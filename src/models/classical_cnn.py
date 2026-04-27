# # from __future__ import annotations

# # import torch
# # from torch import nn


# # class ClassicalCNN(nn.Module):
# #     """1D CNN baseline over UC scenario features.

# #     Input:
# #         x: [batch, feature_dim]
# #     Output:
# #         logits: [batch, num_generators, time_horizon]
# #     """

# #     def __init__(
# #         self,
# #         feature_dim: int,
# #         num_generators: int,
# #         time_horizon: int,
# #         conv_channels: list[int] | tuple[int, ...] = (32, 64, 64),
# #         hidden_dim: int = 256,
# #         dropout: float = 0.20,
# #     ) -> None:
# #         super().__init__()
# #         self.feature_dim = int(feature_dim)
# #         self.num_generators = int(num_generators)
# #         self.time_horizon = int(time_horizon)

# #         layers = []
# #         in_ch = 1
# #         for ch in conv_channels:
# #             layers += [
# #                 nn.Conv1d(in_ch, int(ch), kernel_size=3, padding=1),
# #                 nn.BatchNorm1d(int(ch)),
# #                 nn.ReLU(),
# #                 nn.MaxPool1d(kernel_size=2, stride=2, ceil_mode=True),
# #             ]
# #             in_ch = int(ch)
# #         self.encoder = nn.Sequential(*layers)

# #         with torch.no_grad():
# #             dummy = torch.zeros(1, 1, self.feature_dim)
# #             enc_dim = int(self.encoder(dummy).numel())

# #         self.head = nn.Sequential(
# #             nn.Flatten(),
# #             nn.Linear(enc_dim, hidden_dim),
# #             nn.ReLU(),
# #             nn.Dropout(dropout),
# #             nn.Linear(hidden_dim, self.num_generators * self.time_horizon),
# #         )

# #     def forward(self, x: torch.Tensor) -> torch.Tensor:
# #         z = x.unsqueeze(1)
# #         logits = self.head(self.encoder(z))
# #         return logits.view(-1, self.num_generators, self.time_horizon)
# from __future__ import annotations

# import torch
# from torch import nn


# class ClassicalCNN(nn.Module):
#     """Henderson-style 1D CNN baseline for UC scenario features.

#     This is adapted from the classical CNN baseline used in:
#     Henderson et al., "Quanvolutional Neural Networks: Powering Image Recognition
#     with Quantum Circuits", but modified for 1D UC feature vectors instead of 2D images.

#     Original Henderson-style idea:
#         CONV1 -> POOL1 -> CONV2 -> POOL2 -> FC1 -> DROPOUT -> FC2

#     Input:
#         x: [batch, feature_dim]

#     Output:
#         logits: [batch, num_generators, time_horizon]

#     Notes:
#         - No sigmoid is applied here.
#         - Use BCEWithLogitsLoss during training.
#         - Apply torch.sigmoid(logits) only during inference/evaluation.
#     """

#     def __init__(
#         self,
#         feature_dim: int,
#         num_generators: int,
#         time_horizon: int,
#         conv_channels: list[int] | tuple[int, ...] = (15, 20),
#         hidden_dim: int = 128,
#         dropout: float = 0.40,
#     ) -> None:
#         super().__init__()

#         self.feature_dim = int(feature_dim)
#         self.num_generators = int(num_generators)
#         self.time_horizon = int(time_horizon)

#         # Henderson-style convolutional feature extractor
#         # Adapted from 2D image convolution to 1D UC feature convolution.
#         self.encoder = nn.Sequential(
#             nn.Conv1d(
#                 in_channels=1,
#                 out_channels=int(conv_channels[0]),
#                 kernel_size=5,
#                 stride=1,
#                 padding=2,
#             ),
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=2, stride=2, ceil_mode=True),

#             nn.Conv1d(
#                 in_channels=int(conv_channels[0]),
#                 out_channels=int(conv_channels[1]),
#                 kernel_size=5,
#                 stride=1,
#                 padding=2,
#             ),
#             nn.ReLU(),
#             nn.MaxPool1d(kernel_size=2, stride=2, ceil_mode=True),
#         )

#         # Automatically calculate flattened CNN output size
#         with torch.no_grad():
#             dummy = torch.zeros(1, 1, self.feature_dim)
#             enc_dim = int(self.encoder(dummy).numel())

#         # Henderson-style fully connected head
#         self.head = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(enc_dim, int(hidden_dim)),
#             nn.ReLU(),
#             nn.Dropout(float(dropout)),
#             nn.Linear(
#                 int(hidden_dim),
#                 self.num_generators * self.time_horizon,
#             ),
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         if x.ndim != 2:
#             raise ValueError(
#                 f"Expected input x with shape [batch, feature_dim], got {tuple(x.shape)}"
#             )

#         if x.shape[1] != self.feature_dim:
#             raise ValueError(
#                 f"Expected feature_dim={self.feature_dim}, got x.shape[1]={x.shape[1]}"
#             )

#         # [batch, feature_dim] -> [batch, 1, feature_dim]
#         z = x.unsqueeze(1)

#         # [batch, num_generators * time_horizon]
#         logits = self.head(self.encoder(z))

#         # [batch, num_generators, time_horizon]
#         return logits.view(-1, self.num_generators, self.time_horizon)
from __future__ import annotations

import torch
from torch import nn


class ClassicalCNN(nn.Module):
    """Tiny 1D CNN baseline for UC binary prediction.

    This is the smallest practical CNN:
        Conv1D -> ReLU -> Global Average Pooling -> Linear

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
        conv_channels: int = 8,
        kernel_size: int = 3,
        dropout: float = 0.0,
        hidden_dim:int=0,
    ) -> None:
        super().__init__()

        self.feature_dim = int(feature_dim)
        self.num_generators = int(num_generators)
        self.time_horizon = int(time_horizon)

        padding = kernel_size // 2

        self.encoder = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=int(conv_channels),
                kernel_size=int(kernel_size),
                stride=1,
                padding=padding,
            ),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(float(dropout)),
            nn.Linear(
                int(conv_channels),
                self.num_generators * self.time_horizon,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(
                f"Expected x shape [batch, feature_dim], got {tuple(x.shape)}"
            )

        if x.shape[1] != self.feature_dim:
            raise ValueError(
                f"Expected feature_dim={self.feature_dim}, got {x.shape[1]}"
            )

        z = x.unsqueeze(1)              # [batch, 1, feature_dim]
        logits = self.head(self.encoder(z))

        return logits.view(
            -1,
            self.num_generators,
            self.time_horizon,
        )