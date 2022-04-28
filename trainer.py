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
        "MAE": _AECIFAR10,
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


def _get_model(model_name, z_dim, hidden_dim):
    return {
        "RP": GaussianRandomProjection(n_components=z_dim),
        "PCA": PCA(n_components=z_dim),
        "AE": ndr.AE(z_dim, hidden_dim).cuda(),
        "DAE": ndr.DAE(z_dim, hidden_dim).cuda(),
        "MAE": ndr.MAE(z_dim, hidden_dim).cuda(),
        "VAE": ndr.VAE(z_dim, hidden_dim).cuda(),
        "SimCLR": ndr.SimCLR(z_dim, hidden_dim).cuda(),
    }[model_name]


def _train_model(model, trainloader, n_epochs):
    model.train()
    opt = torch.optim.Adam(model.parameters())
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


def train(model_name, z_dim=256, hidden_dim=64, batch_size=512, n_epochs=10):
    trainloader = _get_loader(batch_size, train=True, model_name=model_name)
    model = _get_model(model_name, z_dim, hidden_dim)
    model = (
        _train_model(model, trainloader, n_epochs)
        if isinstance(model, torch.nn.Module)
        else model.fit(trainloader)
    )
    trainloader = _get_loader(batch_size, train=True)
    testloader = _get_loader(batch_size, train=False)
    return _test_model(model, trainloader, testloader)