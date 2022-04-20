import torch.nn as nn

# https://github.com/nadavbh12/VQ-VAE


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, bn=False):
        super(ResBlock, self).__init__()

        if mid_channels is None:
            mid_channels = out_channels

        layers = [
            nn.ReLU(),
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0),
        ]
        if bn:
            layers.insert(2, nn.BatchNorm2d(out_channels))
        self.convs = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.convs(x)


class Encoder(nn.Sequential):
    def __init__(self, z_dim: int, hidden_dim: int):
        assert z_dim % 64 == 0
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
            # [B, hidden_dim, 8, 8] -> [B, z_dim]
            nn.Conv2d(
                hidden_dim,
                z_dim // 64,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.Flatten(),
        )


class Decoder(nn.Sequential):
    def __init__(self, z_dim: int, hidden_dim: int):
        assert z_dim % 64 == 0
        super().__init__(
            # [B, z_dim] -> [B, hidden_dim, 8, 8]
            nn.Unflatten(1, (z_dim // 64, 8, 8)),
            nn.Conv2d(
                z_dim // 64,
                hidden_dim,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
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
