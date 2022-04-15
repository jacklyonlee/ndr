import numpy as np
import torch
from sklearn.linear_model import SGDClassifier


@torch.no_grad()
def compute_lp(trainloader, testloader, net):
    z_train, y_train = [], []
    for x, y in trainloader:
        x, y = x.cuda(), y.cuda()
        z = net(x)
        z_train.append(z.cpu())
        y_train.append(y.cpu())
    z_train = torch.cat(z_train).numpy()
    y_train = torch.cat(y_train).numpy()
    lp = SGDClassifier(random_state=0).fit(z_train, y_train)

    z_test, y_test = [], []
    for x, y in trainloader:
        x, y = x.cuda(), y.cuda()
        z = net(x)
        z_test.append(z.cpu())
        y_test.append(y.cpu())
    z_test = torch.cat(z_test).numpy()
    y_test = torch.cat(y_test).numpy()
    p_test = lp.predict(z_test)
    acc = np.mean((y_test == p_test).astype(float)) * 100
    print(f"ACC:{acc:.3f}")
