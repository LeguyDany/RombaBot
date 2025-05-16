import torch.nn as nn

class RombaNetwork(nn.Module):
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
        self.value_out = nn.Linear(128, 1)

    def forward(self, x):
        hidden = self.model(x)
        return self.action_out(hidden), self.value_out(hidden)
