import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
import tqdm

import model.metric
import model.ndr

# from sklearn.decomposition import IncrementalPCA
# from sklearn.random_projection import GaussianRandomProjection


BSIZE = 128
NEPOCH = 1


transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

trainset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)

trainloader = data.DataLoader(trainset, batch_size=BSIZE, shuffle=True)

testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)

testloader = data.DataLoader(testset, batch_size=BSIZE, shuffle=False)


def train(net, opt):
    for _ in range(NEPOCH):
        with tqdm.tqdm(trainloader) as t:
            for x, y in t:
                x, y = x.cuda(), y.cuda()
                loss = net.criterion(x)
                opt.zero_grad()
                loss.backward()
                opt.step()
                t.set_description(f"Epoch:{_}/{NEPOCH}|Loss:{loss.item():.2f}")


net = model.ndr.AE(128).cuda()
opt = optim.Adam(net.parameters())

train(net, opt)


def test(net, trainloader, testloader, batch_size):
    # encode dataset with ndr model
    @torch.no_grad()
    def get_features(loader, shuffle=False):
        Z, Y = [], []
        for x, y in tqdm.tqdm(loader):
            if isinstance(net, nn.Module):
                z = net(x.cuda()).cpu().numpy()
            else:  # baseline
                x = torch.flatten(x, start_dim=1).numpy()
                z = net.transform(x)
            Z.append(z)
            Y.append(y)
        return np.concatenate(Z), np.concatenate(Y)

    Z_tr, Y_tr = get_features(trainloader)
    Z_te, Y_te = get_features(testloader)
    model.metric.compute_lp(Z_tr, Y_tr, Z_te, Y_te)
    model.metric.compute_knn(Z_tr, Y_tr, Z_te, Y_te)


test(net, trainloader, testloader, 128)
