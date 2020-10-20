import torch
import torch.nn as nn
import torch.nn.functional as F


# main idea h(state_s)=π(state_s)=argmax(Q(s,a)) ,for all action in state_s

class Net(nn.Module):
    def __init__(self, n_states, n_actions, n_hidden):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(n_states, n_hidden)
        self.out = nn.Linear(n_hidden, n_actions)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)  # ReLU activation
        Q = self.out(x)
        return Q
