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
        return self.fc2(h)


def load_batch():
    return torch.randn(8, 16)


def build_model():
    return Net()


def train_step(model, x):
    y = model(x)
    return y.mean()


def main():
    x = load_batch()
    model = build_model()
    loss = train_step(model, x)
    return loss


if __name__ == "__main__":
    main()
