import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

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

model = nn.Linear(32 * 32 * 3, 10).cuda()
opt = optim.Adam(model.parameters())

for _ in range(NEPOCH):
    for x, y in tqdm.tqdm(trainloader):
        x, y = x.cuda(), y.cuda()
        x = torch.flatten(x, start_dim=1)
        o = model(x)
        loss = F.cross_entropy(o, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

acc = []
for x, y in testloader:
    x, y = x.cuda(), y.cuda()
    x = torch.flatten(x, start_dim=1)
    o = model(x)
    p = o.argmax(axis=1)
    acc.append((p == y).cpu())

acc = torch.cat(acc, axis=0).float().mean()
print("ACC:", acc.item())
