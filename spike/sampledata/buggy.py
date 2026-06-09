import torch
import torch.nn as nn


class Broken(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(64, 4)  # BUG: expects 64 features, gets 32

    def forward(self, x):
        h = self.fc1(x)
        h = torch.relu(h)
        y = self.fc2(h)  # crashes: [8,32] vs Linear(64,4)
        return y


if __name__ == "__main__":
    Broken()(torch.randn(8, 16))
