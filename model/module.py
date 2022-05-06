"""This modules contains implementations of Residual Encoder and Decoder."""

import torch
from torch import nn


class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bn: bool = False,
    ):
        super().__init__()
        layers = [
            nn.ReLU(),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
        ]
        if bn:
            layers.insert(2, nn.BatchNorm2d(out_channels))
        self._convs = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self._convs(x)


class Bottleneck(nn.Sequential):
    def __init__(self, hidden_dim: int):
        super().__init__(
            ResBlock(hidden_dim, hidden_dim, bn=True),
            nn.BatchNorm2d(hidden_dim),
            ResBlock(hidden_dim, hidden_dim, bn=True),
            nn.BatchNorm2d(hidden_dim),
        )


class MLP(nn.Sequential):
    def __init__(self, in_dim: int, out_dim: int):
        hidden_dim = in_dim if in_dim < out_dim else out_dim
        super().__init__(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )


class Encoder(nn.Sequential):
    def __init__(self, n_components: int, hidden_dim: int):
        assert n_components % 64 == 0
        super().__init__(
            nn.Conv2d(
                3,
                hidden_dim // 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden_dim // 2,
                hidden_dim,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            Bottleneck(hidden_dim),
            nn.Flatten(),
            MLP(hidden_dim * 64, n_components),
        )


class Decoder(nn.Sequential):
    def __init__(self, n_components: int, hidden_dim: int):
        assert n_components % 64 == 0
        super().__init__(
            MLP(n_components, hidden_dim * 64),
            nn.Unflatten(1, (hidden_dim, 8, 8)),
            Bottleneck(hidden_dim),
            nn.ConvTranspose2d(
                hidden_dim,
                hidden_dim // 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                hidden_dim // 2,
                3,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
        )
