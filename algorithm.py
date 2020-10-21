import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import pandas as pd
from network import Simple_Net


class DQN:
    def __init__(self,
                 n_states,
                 n_actions,
                 memory_size,
                 replace_target_iter,
                 epsilon=0.15,
                 ep_decay=1e-7,
                 gamma=0.9,
                 lr=0.05,
                 batch_size=32,
                 graph=True
                 ):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        # env
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma

        # memory(s,a,r,s_)
        self.memory_size = memory_size
        self.memory = pd.DataFrame(np.zeros((self.memory_size, n_states*2+2)))

        # greedy
        self.epsilon = epsilon
        self.ep_decay = ep_decay

        # net
        self.target_net, self.eval_net = self.build_net()

        # opt
        self.lr = lr
        self.loss = nn.MSELoss()
        self.optim = optim.Adam(self.eval_net.parameters(), lr=lr)
        self.batch_size = batch_size

        # counter
        self.replace_target_iter = replace_target_iter
        self.learn_step_counter = 0

        # record
        self.cost_history = []

    def build_net(self):
        target_net = Simple_Net(self.n_states, self.n_actions, 20)

        eval_net = Simple_Net(self.n_states, self.n_actions, 20)

        return target_net.to(self.device), eval_net.to(self.device)

    def save_to_memory(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, [a, r], s_))

        # replace old memory to new memory
        index = self.memory_counter % self.memory_size
        self.memory.iloc[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, s):
        # to 2 dimension
        s = torch.tensor(s[np.newaxis, :], dtype=torch.float32).to(self.device)
        # exploitation
        if np.random.uniform() > self.epsilon:
            action_value = self.eval_net(s)
            action = int(action_value.argmax())
        # exploration
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def replace_target_params(self):
        self.target_net.load_state_dict(self.eval_net.state_dict())

    def learn(self):
        # check replace target
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.replace_target_params()
            print('target_net update !')

        # random sampling
        if self.memory_counter > self.memory_size:
            # 記憶超過memoy size 直接作 random sampling
            batch_memory = self.memory.sample(self.batch_size)
        else:
            # 如果記憶次數不及batch_size,重複抽取
            batch_memory = self.memory.iloc[:self.memory_counter].sample(
                self.batch_size, replace=True)
        # catch
        s_ = torch.tensor(
            batch_memory.iloc[:, -self.n_states:].values, dtype=torch.float32)
        s = torch.tensor(
            batch_memory.iloc[:, :self.n_states].values, dtype=torch.float32)
        a = batch_memory.iloc[:, self.n_states:self.n_states+1].values
        r = torch.tensor(
            batch_memory.iloc[:, self.n_states+1:self.n_states+2].values, dtype=torch.float32)

        s, r, s_ = s.to(self.device), r.to(self.device), s_.to(self.device)

        # q
        idx = a.reshape(a.shape[0],)

        q_next = self.target_net(s_).detach()
        q_eval = self.eval_net(s)

        q_esti = q_eval[torch.arange(len(idx)), idx]
        q_target = r.squeeze()+q_next.max(1)[0]

        # cost function
        cost = self.loss(q_target, q_esti)
        self.cost_history.append(cost)

        # backpropagation
        self.optim.zero_grad()
        cost.backward()
        self.optim.step()

        # epsilon decay
        self.epsilon -= self.ep_decay
        self.learn_step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_history)), self.cost_history)
        plt.show()
