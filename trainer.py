import numpy as np
import torch
import torchvision
import tqdm
from PIL import Image
from sklearn.random_projection import GaussianRandomProjection
from sklearnex.decomposition import PCA

import model.metric as metric
import model.ndr as ndr


class _AECIFAR10(torchvision.datasets.CIFAR10):
    def __getitem__(self, index):
        img = self.data[index]
        img = Image.fromarray(img)
        return self.transform(img)


class _SimCLRCIFAR10(torchvision.datasets.CIFAR10):
    def __getitem__(self, index):
        img = self.data[index]
        img = Image.fromarray(img)
        x_i = self.transform(img)
        x_j = self.transform(img)
        return x_i, x_j


def _get_loader(batch_size, train, model_name=None):
    # retrieve dataset
    CIFAR10 = {
        None: torchvision.datasets.CIFAR10,
        "RP": torchvision.datasets.CIFAR10,
        "PCA": torchvision.datasets.CIFAR10,
        "AE": _AECIFAR10,
        "DAE": _AECIFAR10,
        "VAE": _AECIFAR10,
        "SimCLR": _SimCLRCIFAR10,
    }[model_name]

    # define transform
    transform = (
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
        if model_name == "SimCLR"
        else torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
    )

    # make dataset & loader
    dataset = CIFAR10(
        root="./data",
        train=train,
        download=True,
        transform=transform,
    )
    return (
        dataset.data.astype(float).reshape(-1, 3072)
        if model_name in ("RP", "PCA")
        else torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=train,
        )
    )


def _get_model(model_name, n_components, hidden_dim, noise_std, beta):
    return {
        "RP": lambda: GaussianRandomProjection(n_components=n_components),
        "PCA": lambda: PCA(n_components=n_components),
        "AE": lambda: ndr.AE(n_components, hidden_dim).cuda(),
        "DAE": lambda: ndr.DAE(n_components, hidden_dim, noise_std=noise_std).cuda(),
        "VAE": lambda: ndr.VAE(n_components, hidden_dim, beta=beta).cuda(),
        "SimCLR": lambda: ndr.SimCLR(n_components, hidden_dim).cuda(),
    }[model_name]()


def _train_model(model, trainloader, n_epochs):
    model.train()
    opt = torch.optim.AdamW(model.parameters())
    for epoch in range(1, n_epochs + 1):
        with tqdm.tqdm(trainloader) as t:
            for x in t:
                loss = model.criterion(
                    x.cuda() if isinstance(x, torch.Tensor) else [_.cuda() for _ in x]
                )
                opt.zero_grad()
                loss.backward()
                opt.step()
                t.set_description(f"Epoch:{epoch}/{n_epochs}|Loss:{loss.item():.2f}")
    return model.eval()


def _test_model(model, trainloader, testloader):
    @torch.no_grad()
    def get_features(loader):
        # encode dataset with ndr model
        Z, Y = [], []
        for x, y in tqdm.tqdm(loader):
            if isinstance(model, torch.nn.Module):
                z = model(x.cuda()).cpu().numpy()
            else:  # baseline
                x = torch.flatten(x, start_dim=1).numpy()
                z = model.transform(x)
            Z.append(z)
            Y.append(y)
        return np.concatenate(Z), np.concatenate(Y)

    Z_tr, Y_tr = get_features(trainloader)
    Z_te, Y_te = get_features(testloader)
    lp = metric.compute_lp(Z_tr, Y_tr, Z_te, Y_te)
    knn = metric.compute_knn(Z_tr, Y_tr, Z_te, Y_te)
    tsne = metric.compute_tsne(Z_te, Y_te)
    return lp, knn, tsne


def train(
    model_name: str,
    n_components: int = 128,
    hidden_dim: int = 128,
    batch_size: int = 512,
    n_epochs: int = 20,
    noise_std: float = 0.1,
    beta: float = 1e-3,
):
    trainloader = _get_loader(batch_size, train=True, model_name=model_name)
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
    trainloader = _get_loader(batch_size, train=True)
    testloader = _get_loader(batch_size, train=False)
    return _test_model(model, trainloader, testloader)
