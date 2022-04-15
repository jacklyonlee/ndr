import torch.nn as nn
import torch.nn.functional as F

from .module import Decoder, Encoder


class AE(nn.Module):
    def __init__(self, z_dim: int):
        super().__init__()
        self.enc = Encoder(z_dim)
        self.dec = Decoder(z_dim)

    def forward(self, x):
        return self.enc(x)

    def criterion(self, x):
        recon_x = self.dec(self(x))
        return F.mse_loss(recon_x, x)
