import collections
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tqdm

import trainer

sns.color_palette("tab10")
sns.set_theme(style="darkgrid")


def _run_trials(out_dir, filename, n_trials=5, **kwargs):
    results = {"LP": [], "KNN": [], "TSNE": []}
    for _ in range(n_trials):
        lp, knn, tsne = trainer.train(**kwargs)
        print(filename, "LP-ACC", lp, "KNN-ACC", knn)
        results["LP"].append(lp)
        results["KNN"].append(knn)
        results["TSNE"].append(tsne)
    with open(os.path.join(out_dir, filename), "wb+") as f:
        np.save(f, results)


def run_models(out_dir, model_names, n_components, noise_stds, mask_probs, betas):
    # test different n_components
    for model_name in model_names:
        for nc in n_components:
            _run_trials(
                out_dir,
                f"{model_name}-{nc}",
                model_name=model_name,
                n_components=nc,
            )
    # test different noise levels
    for noise_std in noise_stds:
        _run_trials(
            out_dir,
            f"DAE-512-NS{noise_std}",
            model_name="DAE",
            n_components=512,
            noise_std=noise_std,
        )
    # test different mask probs
    for mask_prob in mask_probs:
        _run_trials(
            out_dir,
            f"MAE-512-MP{mask_prob}",
            model_name="MAE",
            n_components=512,
            mask_prob=mask_prob,
        )
    # test different betas
    for beta in betas:
        _run_trials(
            out_dir,
            f"VAE-512-B{beta}",
            model_name="VAE",
            n_components=512,
            beta=beta,
        )


def _get_metrics(out_dir, filename):
    data = np.load(os.path.join(out_dir, filename), allow_pickle=True).item()
    lp = sum(data.get("LP")) / len(data.get("LP"))
    knn = sum(data.get("KNN")) / len(data.get("KNN"))
    tsne = data.get("TSNE")[0]
    return lp, knn, tsne


def _plot_tsne(
    out_dir,
    filename,
    tsne,
    class_names=[
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ],
):
    df = pd.DataFrame(
        {
            "x": tsne[:, 0],
            "y": tsne[:, 1],
            "class": [class_names[int(i)] for i in tsne[:, 2]],
        }
    )
    sns.scatterplot(x="x", y="y", hue="class", data=df)
    plt.savefig(os.path.join(out_dir, f"{filename}-TSNE.png"))
    plt.clf()


def _plot_metric(out_dir, filename, data, id_name, metric, var_name=None):
    sns.lineplot(
        x=id_name,
        y=metric,
        hue=var_name,
        data=pd.DataFrame(data)
        if var_name is None
        else pd.melt(
            pd.DataFrame(data),
            id_vars=[id_name],
            var_name=var_name,
            value_name=metric,
        ),
    )
    plt.savefig(os.path.join(out_dir, f"{filename}-{metric}.png"))
    plt.clf()


def _plot_nc(out_dir, filename, model_names, n_components):
    lp_data = collections.defaultdict(list)
    knn_data = collections.defaultdict(list)
    for nc in tqdm.tqdm(n_components):
        lp_data["n_components"].append(str(nc))
        knn_data["n_components"].append(str(nc))
        for model_name in model_names:
            lp, knn, tsne = _get_metrics(out_dir, f"{model_name}-{nc}")
            lp_data[model_name].append(lp)
            knn_data[model_name].append(knn)
            _plot_tsne(out_dir, f"{model_name}-{nc}", tsne)
    _plot_metric(out_dir, filename, lp_data, "n_components", "LP", "model")
    _plot_metric(out_dir, filename, knn_data, "n_components", "KNN", "model")


def _plot_param(out_dir, filename, param_name, params):
    lp_data = collections.defaultdict(list)
    knn_data = collections.defaultdict(list)
    for param in tqdm.tqdm(params):
        lp_data[param_name].append(str(param))
        knn_data[param_name].append(str(param))
        lp, knn, tsne = _get_metrics(out_dir, f"{filename}{param}")
        lp_data["LP"].append(lp)
        knn_data["KNN"].append(knn)
        _plot_tsne(out_dir, f"{filename}{param}", tsne)
    _plot_metric(out_dir, filename, lp_data, param_name, "LP")
    _plot_metric(out_dir, filename, knn_data, param_name, "KNN")


def plot_models(out_dir, model_names, n_components, noise_stds, mask_probs, betas):
    _plot_nc(out_dir, "NC", model_names, n_components)
    _plot_param(out_dir, "DAE-512-NS", "noise_std", noise_stds)
    _plot_param(out_dir, "MAE-512-MP", "mask_prob", mask_probs)
    _plot_param(out_dir, "VAE-512-B", "beta", betas)


if __name__ == "__main__":
    out_dir, model_names, n_components, noise_stds, mask_probs, betas = (
        "./out",
        ["RP", "PCA", "AE", "DAE", "MAE", "VAE", "SimCLR"],
        [128, 256, 512, 1024],
        [0, 0.1, 0.25, 0.5, 1],
        [0, 0.1, 0.25, 0.5, 0.75],
        [1e-3, 1e-2, 1e-1],
    )
    run_models(out_dir, model_names, n_components, noise_stds, mask_probs, betas)
    plot_models(out_dir, model_names, n_components, noise_stds, mask_probs, betas)
