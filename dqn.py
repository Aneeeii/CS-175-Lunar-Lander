import torch
import torch.nn as nn

class DQN(nn.Module):

    def __init__(self, state_size = 8, action_size = 4): # we change action_size to 6 if we want to do simultaneous engine firing
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLu(),
            nn.Linear(64, 64),
            nn.ReLu(),
            nn.Linear(64, action_size)
        )

    def forward(self, x):
        return self.network(x)