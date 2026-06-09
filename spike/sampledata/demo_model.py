import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, 4)

    def forward(self, x):
        h = self.fc1(x)
        h = torch.relu(h)
        y = self.fc2(h)
        return y


if __name__ == "__main__":
    Net()(torch.randn(8, 16))
