"""This module contains implementations of AE, DAE, VAE and SimCLR."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import Decoder, Encoder


class AE(nn.Module):
    def __init__(self, n_components: int, hidden_dim: int):
        super().__init__()
        self._enc = Encoder(n_components, hidden_dim)
        self._dec = Decoder(n_components, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._enc(x)

    def criterion(self, x: torch.Tensor) -> torch.Tensor:
        recon_x = self._dec(self(x))
        return F.mse_loss(recon_x, x)


def _perturb(x: torch.Tensor, std: float) -> torch.Tensor:
    x_p = x + torch.empty_like(x).normal_(0, std)
    return F.normalize(x_p, dim=1)


class DAE(AE):
    def __init__(
        self,
        n_components: int,
        hidden_dim: int,
        sigma: float = 0.1,
    ):
        super().__init__(n_components, hidden_dim)
        self._sigma = sigma

    def criterion(self, x: torch.Tensor) -> torch.Tensor:
        z = self(_perturb(x, self._sigma))
        recon_x = self._dec(z)
        return F.mse_loss(recon_x, x)


def _reparameterize(
    z: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mu, logvar = torch.chunk(z, 2, dim=-1)
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return eps * std + mu, mu, logvar


class VAE(nn.Module):
    def __init__(
        self,
        n_components: int,
        hidden_dim: int,
        beta: float = 1e-3,
    ):
        super().__init__()
        self._enc = Encoder(n_components * 2, hidden_dim)
        self._dec = Decoder(n_components, hidden_dim)
        self._beta = beta

    def forward(self, x):
        return _reparameterize(self._enc(x))[0]

    def criterion(self, x: torch.Tensor) -> torch.Tensor:
        z, mu, logvar = _reparameterize(self._enc(x))
        recon_x = self._dec(z)
        recon_loss = F.mse_loss(recon_x, x)
        kl_loss = ((mu**2 + logvar.exp() - 1 - logvar) / 2).mean()
        return recon_loss + self._beta * kl_loss


class SimCLR(nn.Module):
    def __init__(
        self,
        n_components: int,
        hidden_dim: int,
        proj_dim: int = 128,
        tau: float = 0.5,
    ):
        super().__init__()
        self._enc = Encoder(n_components, hidden_dim)
        self._proj = nn.Sequential(
            nn.BatchNorm1d(n_components),
            nn.ReLU(inplace=True),
            nn.Linear(n_components, proj_dim),
        )
        self._tau = tau

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._enc(x)

    def criterion(
        self,
        x: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        z_i, z_j = (self._proj(self(_)) for _ in x)
        z = torch.cat((z_i, z_j), dim=0)
        # compute similarity matrix
        sim = (z @ z.T) / (
            torch.linalg.norm(z, dim=1, keepdim=True)
            @ torch.linalg.norm(z.T, dim=0, keepdim=True)
        )
        # compute denominator
        exp = torch.exp(sim / self._tau)
        mask = torch.ones_like(exp) - torch.eye(z.size(0)).to(z)
        exp = exp.masked_select(mask.bool()).view(z.size(0), -1)
        denom = exp.sum(dim=1, keepdim=True)
        # compute positive pairs
        pos = (z_i * z_j).sum(dim=1, keepdim=True) / (
            torch.linalg.norm(z_i, dim=1, keepdim=True)
            * torch.linalg.norm(z_j, dim=1, keepdim=True)
        )
        # compute numerator
        num = torch.exp(pos / self._tau).repeat(2, 1)
        return -torch.log(num / denom).sum() / z.size(0)
