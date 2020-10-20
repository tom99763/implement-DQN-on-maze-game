import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F

from utils import ReplayMemory
from collections import namedtuple
import random
from network import Net

import numpy as np
import random


class DQN(ReplayMemory):
    def __init__(self,
                 n_states,
                 n_actions,
                 n_hidden,
                 memory_capacity,
                 batch_size,
                 target_replace_iter,
                 epsilon=0.2,
                 lr=0.01,
                 gamma=0.9,
                 decay=1e-6
                 ):
        super(DQN, self).__init__(memory_capacity)
        # device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.experience = namedtuple('Experience', ('s', 'a', 'r', 's_'))
        self.batch_size = batch_size
        self.memory_capacity = memory_capacity

        # env
        self.n_states = n_states
        self.n_actions = n_actions

        # greedy
        self.epsilon = epsilon
        self.decay = decay

        # net
        self.value_net = Net(n_states, n_actions, n_hidden).to(self.device)
        self.target_net = Net(n_states, n_actions, n_hidden).to(self.device)
        # initialize net
        self.target_net.load_state_dict(self.value_net.state_dict())

        # optim
        self.optim = Adam(self.value_net.parameters(), lr=lr)
        self.mse = nn.MSELoss()

        # record step
        self.learn_step_counter = 0
        self.target_replace_iter = target_replace_iter
        self.gamma = gamma

    # epsilon greedy method
    def choose_action(self, s):
        x = torch.unsqueeze(torch.tensor(
            s, dtype=torch.float32), 0).to(self.device)

        # exploration
        if np.random.uniform() < self.epsilon:  # 隨機
            action = np.random.randint(0, self.n_actions)
            self.epsilon -= self.decay

        # exploitation
        else:
            Q_hat = self.value_net(x)  # 以現有 eval net 得出各個 action 的分數
            action = torch.argmax(Q_hat, 1)  # 挑選最高分的 action
        return int(action)

    def algorithm(self):

        # random_sampling
        random_sample = self.sample(self.batch_size)
        # to tensor
        r_s, r_a, r_r, r_s_ = self.extract_tensors(random_sample)  # r:random
        r_s, r_a, r_r, r_s_ = r_s.to(self.device), r_a, r_r.to(
            self.device), r_s_.to(self.device)
        # evaluate Q(s,a) ,Q(s_,s_)
        idx = list(r_a.data.numpy().astype(int))
        q_value = self.value_net(r_s)
        q_value = q_value[torch.arange(len(idx)), idx]
        q_next = self.target_net(r_s_).detach()

        # Q target
        q_target = r_r+self.gamma*q_next.max(1)[0]

        # loss
        loss = self.mse(q_value, q_target)

        # backpropogation
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        # check_target_net_update,fixed target point
        # check whether updating target net
        self.check_target_net_update()

    # update:copy value net weight and add into target net

    def check_target_net_update(self):
        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_replace_iter == 0:
            self.target_net.load_state_dict(
                self.value_net.state_dict())  # fixed point

    # save (s,a,r,s_) to experience
    def save_memory(self, s, a, r, s_):
        e = self.experience(torch.tensor([s]),
                            torch.tensor([a]),
                            torch.tensor([r]),
                            torch.tensor([s_]))
        self.push(e)
