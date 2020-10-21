import torch
import torch.nn as nn
import torch.nn.functional as F


# main idea h(state_s)=π(state_s)=argmax(Q(s,a)) ,for all action in state_s

class Net(nn.Module):
    def __init__(self, n_states, n_actions, n_hidden):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(n_states, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden*2)
        self.fc3 = nn.Linear(n_hidden*2, n_hidden)
        self.out = nn.Linear(n_hidden, n_actions)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)  # ReLU activation
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        Q = self.out(x)
        return Q


class Simple_Net(nn.Module):
    def __init__(self, n_states, n_actions, n_hidden):
        super(Simple_Net, self).__init__()

        self.fc1 = nn.Linear(n_states, n_hidden)
        self.out = nn.Linear(n_hidden, n_actions)

    def forward(self, x):
        x = self.fc1(x)
        Q = self.out(x)
        return Q
