from typing import Optional, Tuple, Union

import numpy as np
import sklearn.base as base
import sklearn.random_projection as random_projection
import sklearnex.decomposition as decomposition
import torch
import torch.nn as nn
import torchvision
import tqdm
from PIL import Image
from torch.utils.data.dataloader import DataLoader

import model.metric as metric
import model.ndr as ndr


class _AECIFAR10(torchvision.datasets.CIFAR10):
    def __getitem__(self, index: int) -> torch.Tensor:
        img = self.data[index]
        img = Image.fromarray(img)
        return self.transform(img)


class _SimCLRCIFAR10(torchvision.datasets.CIFAR10):
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img = self.data[index]
        img = Image.fromarray(img)
        x_i = self.transform(img)
        x_j = self.transform(img)
        return x_i, x_j


def _get_transform(is_simclr: bool) -> torchvision.transforms.Compose:
    return (
        torchvision.transforms.Compose(
            [
                # Randomly resize and crop to 32x32.
                torchvision.transforms.RandomResizedCrop(32),
                # Horizontally flip the image with probability 0.5
                torchvision.transforms.RandomHorizontalFlip(p=0.5),
                # With a probability of 0.8, apply color jitter
                torchvision.transforms.RandomApply(
                    [torchvision.transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8
                ),
                # With a probability of 0.2, convert the image to grayscale
                torchvision.transforms.RandomGrayscale(p=0.2),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        if is_simclr
        else torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
    )


def _get_loader(
    batch_size: int,
    train: bool,
    model_name: Optional[str] = None,
) -> Union[np.ndarray, DataLoader]:
    dataset = {
        None: torchvision.datasets.CIFAR10,
        "rp": torchvision.datasets.CIFAR10,
        "pca": torchvision.datasets.CIFAR10,
        "ae": _AECIFAR10,
        "dae": _AECIFAR10,
        "vae": _AECIFAR10,
        "simclr": _SimCLRCIFAR10,
    }[model_name](
        root="./data",
        train=train,
        download=True,
        transform=_get_transform(model_name == "simclr"),
    )
    return (
        dataset.data.astype(float).reshape(-1, 3072)
        if model_name in ("rp", "pca")
        else DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=train,
        )
    )


def _get_model(
    model_name: str,
    n_components: int,
    hidden_dim: int,
    noise_std: float,
    beta: float,
) -> Union[base.BaseEstimator, nn.Module]:
    return {
        "rp": lambda: random_projection.GaussianRandomProjection(n_components),
        "pca": lambda: decomposition.PCA(n_components),
        "ae": lambda: ndr.AE(n_components, hidden_dim).cuda(),
        "dae": lambda: ndr.DAE(n_components, hidden_dim, noise_std).cuda(),
        "vae": lambda: ndr.VAE(n_components, hidden_dim, beta).cuda(),
        "simclr": lambda: ndr.SimCLR(n_components, hidden_dim).cuda(),
    }[model_name]()


def _train_model(
    model: nn.Module,
    trainloader: DataLoader,
    n_epochs: int,
) -> nn.Module:
    model.train()
    opt = torch.optim.AdamW(model.parameters())
    for epoch in range(1, n_epochs + 1):
        with tqdm.tqdm(trainloader) as t:
            for x in t:
                loss = model.criterion(
                    x.cuda()
                    if isinstance(x, torch.Tensor)
                    else tuple(_.cuda() for _ in x)
                )
                opt.zero_grad()
                loss.backward()
                opt.step()
                t.set_description(f"Epoch:{epoch}/{n_epochs}|Loss:{loss.item():.2f}")
    return model.eval()


def _test_model(
    model: Union[base.BaseEstimator, nn.Module],
    trainloader: DataLoader,
    testloader: DataLoader,
) -> Tuple[float, float, np.ndarray]:
    @torch.no_grad()
    def get_features(loader: DataLoader):
        # encode dataset with ndr model
        z_, y_ = [], []
        for x, y in tqdm.tqdm(loader):
            if isinstance(model, torch.nn.Module):
                z = model(x.cuda()).cpu().numpy()
            else:  # baseline
                x = torch.flatten(x, start_dim=1).numpy()
                z = model.transform(x)
            z_.append(z)
            y_.append(y)
        return np.concatenate(z_), np.concatenate(y_)

    z_tr, y_tr = get_features(trainloader)
    z_te, y_te = get_features(testloader)
    lp = metric.compute_lp(z_tr, y_tr, z_te, y_te)
    knn = metric.compute_knn(z_tr, y_tr, z_te, y_te)
    tsne = metric.compute_tsne(z_te, y_te)
    return lp, knn, tsne


def train(
    model_name: str,
    n_components: int = 512,
    hidden_dim: int = 128,
    batch_size: int = 512,
    n_epochs: int = 20,
    noise_std: float = 0.1,
    beta: float = 1e-3,
) -> Tuple[float, float, np.ndarray]:
    trainloader = _get_loader(
        batch_size,
        train=True,
        model_name=model_name,
    )
    model = _get_model(
        model_name,
        n_components,
        hidden_dim,
        noise_std=noise_std,
        beta=beta,
    )
    model = (
        _train_model(model, trainloader, n_epochs)
        if isinstance(model, torch.nn.Module)
        else model.fit(trainloader)
    )
    return _test_model(
        model,
        _get_loader(batch_size, train=True),
        _get_loader(batch_size, train=False),
    )
