"""This module contains functions to train and test models using CIFAR10."""

from typing import Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.base import BaseEstimator
from sklearn.random_projection import GaussianRandomProjection
from sklearnex.decomposition import PCA
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from tqdm import tqdm

from model.metric import compute_knn, compute_lp, compute_tsne
from model.ndr import AE, DAE, VAE, SimCLR


class _AECIFAR10(CIFAR10):
    def __getitem__(self, index: int) -> torch.Tensor:
        img = self.data[index]
        img = Image.fromarray(img)
        return self.transform(img)


class _SimCLRCIFAR10(CIFAR10):
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        img = self.data[index]
        img = Image.fromarray(img)
        x_i = self.transform(img)
        x_j = self.transform(img)
        return x_i, x_j


def _get_transform(is_simclr: bool) -> transforms.Compose:
    return (
        transforms.Compose(
            [
                # Randomly resize and crop to 32x32.
                transforms.RandomResizedCrop(32),
                # Horizontally flip the image with probability 0.5
                transforms.RandomHorizontalFlip(p=0.5),
                # With a probability of 0.8, apply color jitter
                transforms.RandomApply(
                    [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8
                ),
                # With a probability of 0.2, convert the image to grayscale
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        if is_simclr
        else transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
    )


def _get_loader(
    batch_size: int,
    is_train: bool,
    model: str = "",
) -> DataLoader:
    return DataLoader(
        {
            **dict.fromkeys(["", "rp", "pca"], CIFAR10),
            **dict.fromkeys(["ae", "dae", "vae"], _AECIFAR10),
            "simclr": _SimCLRCIFAR10,
        }[model](
            root="./data",
            train=is_train,
            download=True,
            transform=_get_transform(model == "simclr"),
        ),
        batch_size=batch_size,
        shuffle=is_train,
    )


def _get_model(
    model: str,
    n_components: int,
    hidden_dim: int,
    sigma: float,
    beta: float,
) -> Union[BaseEstimator, nn.Module]:
    return {
        "rp": lambda: GaussianRandomProjection(n_components),
        "pca": lambda: PCA(n_components),
        "ae": lambda: AE(n_components, hidden_dim),
        "dae": lambda: DAE(n_components, hidden_dim, sigma),
        "vae": lambda: VAE(n_components, hidden_dim, beta),
        "simclr": lambda: SimCLR(n_components, hidden_dim),
    }[model]()


def _train_model(
    net: nn.Module,
    trainloader: DataLoader,
    n_epochs: int,
):
    net.cuda().train()
    opt = torch.optim.AdamW(net.parameters())
    for _ in range(n_epochs):
        with tqdm(trainloader) as t:
            for x in t:
                loss = net.criterion(
                    x.cuda()
                    if isinstance(x, torch.Tensor)
                    else tuple(_.cuda() for _ in x)
                )
                opt.zero_grad()
                loss.backward()
                opt.step()
                t.set_description(f"E:{_+1}/{n_epochs}|L:{loss.item():.2f}")


@torch.no_grad()
def _get_features(
    net: Union[BaseEstimator, nn.Module],
    loader: DataLoader,
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(net, nn.Module):
        net.cuda().eval()
    z_list, y_list = [], []
    for x, y in tqdm(loader):
        if isinstance(net, nn.Module):
            z = net(x.cuda()).cpu().numpy()
        else:
            x = torch.flatten(x, start_dim=1).numpy()
            z = net.transform(x)
        z_list.append(z)
        y_list.append(y)
    return np.concatenate(z_list), np.concatenate(y_list)


def _test_model(
    net: Union[BaseEstimator, nn.Module],
    trainloader: DataLoader,
    testloader: DataLoader,
) -> tuple[float, float, np.ndarray]:
    z_tr, y_tr = _get_features(net, trainloader)
    z_te, y_te = _get_features(net, testloader)
    return (
        compute_lp(z_tr, y_tr, z_te, y_te),
        compute_knn(z_tr, y_tr, z_te, y_te),
        compute_tsne(z_te, y_te),
    )


def train(
    model: str,
    n_components: int = 128,
    hidden_dim: int = 128,
    batch_size: int = 512,
    n_epochs: int = 20,
    sigma: float = 0.1,
    beta: float = 1e-3,
) -> tuple[float, float, np.ndarray]:
    """Performs training and testing of specified model.

    Parameters
    ----------
    model
        Model to be trained. Supports Random Projection (rp),
        Principle Component Analysis (pca), Autoencoder (ae),
        Denosing Autoencoder (dae), Variantional Autoencoder (vae) and
        Contrastive Learning (simclr).
    n_components
        Dimensionality reduction feature dimension.
    hidden_dim
        Number of hidden channels for model.
    batch_size
        Batch size for training and testing.
    n_epochs
        Number of epochs for training.
    sigma
        Noise standard deviation for Denosing Autoencoder.
    beta
        Beta value for Variantional Autoencoder.

    Returns
    -------
    tuple[float, float, np.ndarray]
        Linear Probe accuracy, Nearest Neighbor accuracy, t-SNE embeddings.
    """
    trainloader = _get_loader(
        batch_size,
        is_train=True,
        model=model,
    )
    net = _get_model(
        model,
        n_components,
        hidden_dim,
        sigma=sigma,
        beta=beta,
    )
    (
        _train_model(net, trainloader, n_epochs)
        if isinstance(net, nn.Module)
        else net.fit(trainloader.dataset.data.reshape(-1, 3072))
    )
    return _test_model(
        net,
        _get_loader(batch_size, is_train=True),
        _get_loader(batch_size, is_train=False),
    )


if __name__ == "__main__":
    lp, knn, _ = train("ae", n_components=128, n_epochs=10)
    print(f"lp-acc: {lp} knn-acc: {knn}")
