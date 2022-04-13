import torch.nn.functional as F


class AELoss:
    def __call__(self, recon_x, x):
        return F.mse_loss(recon_x, x)
