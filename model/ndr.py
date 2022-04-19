import torch
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


class VAE(nn.Module):
    def __init__(self, z_dim: int, beta: float):
        super().__init__()
        self.enc = Encoder(z_dim * 2)
        self.dec = Decoder(z_dim)
        self.beta = beta

    def _reparameterize(self, z):
        mu, logvar = torch.chunk(z, 2, dim=-1)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu, mu, logvar

    def forward(self, x):
        return self._reparameterize(self.enc(x))[0]

    def criterion(self, x):
        z, mu, logvar = self._reparameterize(self.enc(x))
        recon_x = self.dec(z)
        recon_loss = F.mse_loss(recon_x, x)
        kl_loss = ((mu**2 + logvar.exp() - 1 - logvar) / 2).mean()
        return recon_loss + self.beta * kl_loss
