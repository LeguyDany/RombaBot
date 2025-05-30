import torch.nn as nn

class PolicyNetwork(nn.Module):
    def __init__(self, obs_size, action_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(obs_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.action_out = nn.Linear(128, action_size)

    def forward(self, x):
        hidden = self.model(x)
        return self.action_out(hidden)
