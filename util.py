import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
from sklearn.random_projection import GaussianRandomProjection
from sklearnex.decomposition import PCA

import model.ndr as ndr


class _ReconCIFAR10(datasets.CIFAR10):
    def __getitem__(self, index):
        img = self.data[index]
        img = Image.fromarray(img)
        x = self.transform(img)
        return x


class _ContrastiveCIFAR10(datasets.CIFAR10):
    def __getitem__(self, index):
        img = self.data[index]
        img = Image.fromarray(img)
        x_i = self.transform(img)
        x_j = self.transform(img)
        return x_i, x_j


def get_dataloader(root, batch_size, train, model_name=None):
    # retrieve dataset
    CIFAR10 = {
        None: datasets.CIFAR10,
        "RP": datasets.CIFAR10,
        "PCA": datasets.CIFAR10,
        "AE": _ReconCIFAR10,
        "VAE": _ReconCIFAR10,
        "SimCLR": _ContrastiveCIFAR10,
    }[model_name]
    # define transform
    transform = (
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
        if model_name == "SimCLR"
        else transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
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
    if model_name in ("RP", "PCA"):
        return dataset.data.astype(float).reshape(-1, 3072)
    return data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
    )


def get_model(model_name, z_dim, hidden_dim):
    return {
        "RP": GaussianRandomProjection(n_components=z_dim),
        "PCA": PCA(n_components=z_dim),
        "AE": ndr.AE(z_dim, hidden_dim).cuda(),
        "VAE": ndr.VAE(z_dim, hidden_dim).cuda(),
    }[model_name]
