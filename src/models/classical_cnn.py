"""Stage 3A placeholder: classical CNN baseline for UC commitment prediction."""

from __future__ import annotations

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover
    torch = None
    nn = None


class ClassicalUCCommitmentCNN(nn.Module if nn is not None else object):
    """Scaffold for a CNN predicting [n_generators, time_horizon] binaries."""

    def __init__(self, input_dim: int, n_generators: int, time_horizon: int):
        if nn is None:
            raise ImportError("Install torch to use ClassicalUCCommitmentCNN")
        super().__init__()
        self.input_dim = input_dim
        self.n_generators = n_generators
        self.time_horizon = time_horizon
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, n_generators * time_horizon),
        )

    def forward(self, x):
        logits = self.net(x)
        return logits.view(-1, self.n_generators, self.time_horizon)
