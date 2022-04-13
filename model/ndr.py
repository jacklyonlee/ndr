import torch.nn as nn

from .module import Decoder, Encoder


class AE(nn.Module):
    def __init__(self, z_dim: int):
        super().__init__()
        self.enc = Encoder(z_dim)
        self.dec = Decoder(z_dim)

    def forward(self, x):
        z = self.enc(x)
        recon_x = self.dec(z)
        return recon_x
