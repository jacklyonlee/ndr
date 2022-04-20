import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm

import model.metric as metric
import util


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help=(
            "Dimensionality reduction model "
            "(supported models: RP, PCA, AE, VAE, SimCLR)."
        ),
    )
    parser.add_argument(
        "--z_dim", type=int, required=True, help="Dimension after reduction."
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=64, help="Model hidden dimension."
    )
    parser.add_argument(
        "--batch_size", type=int, default=512, help="Training batch size."
    )
    parser.add_argument(
        "--n_epochs", type=int, default=1, help="Number of epochs to train."
    )
    return parser.parse_args()


def main(model_name, z_dim, hidden_dim, batch_size, n_epochs):
    # prepare training data
    trainloader = util.get_dataloader(
        root="./data", model_name=model_name, batch_size=batch_size, train=True
    )

    # select model
    model = util.get_model(model_name, z_dim, hidden_dim)

    def train(model, loader):
        model.train()
        opt = optim.Adam(model.parameters())
        for _ in range(n_epochs):
            with tqdm.tqdm(loader) as t:
                for x in t:
                    loss = model.criterion(x.cuda())
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    t.set_description(f"Epoch:{_}/{n_epochs}|Loss:{loss.item():.2f}")
        model.eval()

    # train model
    if isinstance(model, nn.Module):
        train(model, trainloader)
    else:
        model.fit(trainloader)

    # prepare test data
    del trainloader
    trainloader = util.get_dataloader(root="./data", batch_size=batch_size, train=True)
    testloader = util.get_dataloader(root="./data", batch_size=batch_size, train=False)

    def test(model, trainloader, testloader):
        # encode dataset with ndr model
        @torch.no_grad()
        def get_features(loader):
            Z, Y = [], []
            for x, y in tqdm.tqdm(loader):
                if isinstance(model, nn.Module):
                    z = model(x.cuda()).cpu().numpy()
                else:  # baseline
                    x = torch.flatten(x, start_dim=1).numpy()
                    z = model.transform(x)
                Z.append(z)
                Y.append(y)
            return np.concatenate(Z), np.concatenate(Y)

        Z_tr, Y_tr = get_features(trainloader)
        Z_te, Y_te = get_features(testloader)
        metric.compute_lp(Z_tr, Y_tr, Z_te, Y_te)
        metric.compute_knn(Z_tr, Y_tr, Z_te, Y_te)

    # test model
    test(model, trainloader, testloader)


if __name__ == "__main__":
    args = parse_args()
    main(args.model, args.z_dim, args.hidden_dim, args.batch_size, args.n_epochs)
