from __future__ import annotations

import torch
from torch import nn


class ClassicalCNN(nn.Module):
    """1D CNN baseline over UC scenario features.

    Input:
        x: [batch, feature_dim]
    Output:
        logits: [batch, num_generators, time_horizon]
    """

    def __init__(
        self,
        feature_dim: int,
        num_generators: int,
        time_horizon: int,
        conv_channels: list[int] | tuple[int, ...] = (32, 64, 64),
        hidden_dim: int = 256,
        dropout: float = 0.20,
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.num_generators = int(num_generators)
        self.time_horizon = int(time_horizon)

        layers = []
        in_ch = 1
        for ch in conv_channels:
            layers += [
                nn.Conv1d(in_ch, int(ch), kernel_size=3, padding=1),
                nn.BatchNorm1d(int(ch)),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2, ceil_mode=True),
            ]
            in_ch = int(ch)
        self.encoder = nn.Sequential(*layers)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.feature_dim)
            enc_dim = int(self.encoder(dummy).numel())

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(enc_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_generators * self.time_horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x.unsqueeze(1)
        logits = self.head(self.encoder(z))
        return logits.view(-1, self.num_generators, self.time_horizon)
