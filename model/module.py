import torch
import torch.nn as nn

# https://github.com/nadavbh12/VQ-VAE


class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bn: bool = False,
    ) -> None:
        super(ResBlock, self).__init__()
        layers = [
            nn.ReLU(),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
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
        self.convs = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.convs(x)


class Encoder(nn.Sequential):
    def __init__(self, n_components: int, hidden_dim: int) -> None:
        assert n_components % 64 == 0
        super().__init__(
            # downsample
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
            # residual bottleneck
            ResBlock(hidden_dim, hidden_dim, bn=True),
            nn.BatchNorm2d(hidden_dim),
            ResBlock(hidden_dim, hidden_dim, bn=True),
            # [B, hidden_dim, 8, 8] -> [B, n_components]
            nn.Flatten(),
            nn.Linear(hidden_dim * 64, n_components, bias=False),
            nn.BatchNorm1d(n_components),
            nn.ReLU(inplace=True),
            nn.Linear(n_components, n_components, bias=True),
        )


class Decoder(nn.Sequential):
    def __init__(self, n_components: int, hidden_dim: int) -> None:
        assert n_components % 64 == 0
        super().__init__(
            # [B, n_components] -> [B, hidden_dim, 8, 8]
            nn.Linear(n_components, n_components, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(n_components),
            nn.Linear(n_components, hidden_dim * 64, bias=False),
            nn.Unflatten(1, (hidden_dim, 8, 8)),
            # residual bottleneck
            ResBlock(hidden_dim, hidden_dim, bn=True),
            nn.BatchNorm2d(hidden_dim),
            ResBlock(hidden_dim, hidden_dim, bn=True),
            nn.BatchNorm2d(hidden_dim),
            # image reconstruction
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
