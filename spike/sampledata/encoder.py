"""A PURE LIBRARY model — defines a module and runs nothing on import.

There is no `if __name__ == "__main__"` block, so "▶ Trace this file" finds no
shapes (nothing executes). This is the common real-world case: a model class you
import elsewhere. Click the per-function "▶ trace" on `forward` to call it
directly with an auto-synthesized input — no __main__, no debugger. The cockpit
shows the exact call it ran (e.g. `Encoder().forward(randn(2, 16))`) so the
shapes are reproducible and the input is never hidden.
"""
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, d_in=16, d_hidden=64, d_out=8):
        super().__init__()
        self.proj = nn.Linear(d_in, d_hidden)   # first Linear -> synth input randn(2, 16)
        self.norm = nn.LayerNorm(d_hidden)
        self.out = nn.Linear(d_hidden, d_out)

    def forward(self, x):
        h = self.proj(x)            # [2, 16] -> [2, 64]
        h = torch.relu(self.norm(h))
        z = self.out(h)             # [2, 64] -> [2, 8]
        return z
