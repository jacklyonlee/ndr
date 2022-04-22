import torch
import torch.nn as nn
import torch.nn.functional as F

from .module import Decoder, Encoder


class AE(nn.Module):
    def __init__(self, z_dim: int, hidden_dim: int):
        super().__init__()
        self.enc = Encoder(z_dim, hidden_dim)
        self.dec = Decoder(z_dim, hidden_dim)

    def forward(self, x):
        return self.enc(x)

    def criterion(self, x):
        recon_x = self.dec(self(x))
        return F.mse_loss(recon_x, x)


class DAE(AE):
    def __init__(self, z_dim: int, hidden_dim: int, noise_std: float = 0.1):
        super().__init__(z_dim, hidden_dim)
        self.noise_std = noise_std

    def _perturb(self, x):
        x_p = x + torch.empty_like(x).normal_(0, self.noise_std)
        return F.normalize(x_p, dim=1)

    def criterion(self, x):
        z = self(self._perturb(x))
        recon_x = self.dec(z)
        return F.mse_loss(recon_x, x)


class MAE(AE):
    def __init__(self, z_dim: int, hidden_dim: int, mask_prob: float = 0.25):
        super().__init__(z_dim, hidden_dim)
        self.mask_prob = mask_prob

    def _perturb(self, x):
        return x * (torch.rand_like(x) > self.mask_prob)

    def criterion(self, x):
        z = self(self._perturb(x))
        recon_x = self.dec(z)
        return F.mse_loss(recon_x, x)


class VAE(nn.Module):
    def __init__(self, z_dim: int, hidden_dim: int, beta: float = 1e-3):
        super().__init__()
        self.enc = Encoder(z_dim * 2, hidden_dim)
        self.dec = Decoder(z_dim, hidden_dim)
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


class SimCLR(nn.Module):
    def __init__(
        self, z_dim: int, hidden_dim: int, proj_dim: int = 128, tau: float = 0.5
    ):
        super().__init__()
        self.enc = Encoder(z_dim, hidden_dim)
        self.proj = nn.Sequential(
            nn.Linear(z_dim, z_dim, bias=False),
            nn.BatchNorm1d(z_dim),
            nn.ReLU(inplace=True),
            nn.Linear(z_dim, proj_dim, bias=True),
        )
        self.tau = tau

    def forward(self, x):
        return self.enc(x)

    def criterion(self, x):
        z_i = self.proj(self(x[0]))
        z_j = self.proj(self(x[1]))
        z = torch.cat([z_i, z_j], dim=0)
        # compute similarity matrix
        sim = (z @ z.T) / (
            torch.linalg.norm(z, dim=1, keepdim=True)
            @ torch.linalg.norm(z.T, dim=0, keepdim=True)
        )
        # compute denominator
        exp = torch.exp(sim / self.tau)
        mask = (torch.ones_like(exp) - torch.eye(z.size(0)).to(exp)).bool()
        exp = exp.masked_select(mask).view(z.size(0), -1)
        denom = exp.sum(dim=1, keepdim=True)
        # compute positive pairs
        pos = (z_i * z_j).sum(dim=1, keepdim=True) / (
            torch.linalg.norm(z_i, dim=1, keepdim=True)
            * torch.linalg.norm(z_j, dim=1, keepdim=True)
        )
        # compute numerator
        num = torch.exp(pos / self.tau).repeat(2, 1)
        return (-torch.log(num / denom)).sum() / z.size(0)
