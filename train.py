import torch
import torch.optim as optim
import torchvision
import tqdm

import model.metric
import model.ndr

BSIZE = 64
NEPOCH = 2


transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

trainset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=BSIZE, shuffle=True)

testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)

testloader = torch.utils.data.DataLoader(testset, batch_size=BSIZE, shuffle=False)


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
model.metric.compute_lp(trainloader, testloader, net)
