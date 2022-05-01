import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tqdm

import trainer

sns.color_palette("deep")
sns.set_theme(style="darkgrid")


def _run_trials(out_dir, filename, n_trials=5, **kwargs):
    results = {"lp": [], "knn": [], "tsne": []}
    for _ in range(n_trials):
        lp, knn, tsne = trainer.train(**kwargs)
        print(filename, "lp-acc", lp, "knn-acc", knn)
        results["lp"].append(lp)
        results["knn"].append(knn)
        results["tsne"].append(tsne)
    with open(os.path.join(out_dir, f"{filename}.npy"), "wb+") as f:
        np.save(f, results)


def run_models(out_dir, model_names, n_components, noise_stds, mask_probs, betas):
    # test different n_components
    for model_name in model_names:
        for n_c in n_components:
            _run_trials(
                out_dir,
                f"{model_name}-{n_c}",
                model_name=model_name,
                n_components=n_c,
            )
    # test different noise levels
    for noise_std in noise_stds:
        _run_trials(
            out_dir,
            f"DAE-128-noise{noise_std}",
            model_name="DAE",
            n_components=128,
            noise_std=noise_std,
        )
    # test different mask probs
    for mask_prob in mask_probs:
        _run_trials(
            out_dir,
            f"MAE-128-mask{mask_prob}",
            model_name="MAE",
            n_components=128,
            mask_prob=mask_prob,
        )
    # test different betas
    for beta in betas:
        _run_trials(
            out_dir,
            f"VAE-128-beta{beta}",
            model_name="VAE",
            n_components=128,
            beta=beta,
        )


def _get_metrics(out_dir, filename):
    data = np.load(
        os.path.join(out_dir, filename),
        allow_pickle=True,
    ).item()
    lp = data.get("lp")
    knn = data.get("knn")
    tsne = data.get("tsne")[0]
    return lp, knn, tsne


def _plot_fig(out_dir, filename):
    plt.legend(
        bbox_to_anchor=(1.02, 1),
        loc=2,
        borderaxespad=0,
        frameon=False,
    )
    plt.savefig(
        os.path.join(out_dir, f"{filename}.png"),
        bbox_inches="tight",
    )
    plt.clf()


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
    sns.scatterplot(
        x="x",
        y="y",
        hue="class",
        linewidth=0,
        data=pd.DataFrame(
            {
                "x": tsne[:, 0],
                "y": tsne[:, 1],
                "class": [class_names[int(i)] for i in tsne[:, 2]],
            }
        ),
    )
    _plot_fig(out_dir, f"{filename}-tsne")


def _plot_metric(out_dir, filename, data, x_name, y_name, hue_name):
    sns.lineplot(
        x=x_name,
        y=y_name,
        hue=hue_name,
        ci=95,
        data=pd.DataFrame(
            data,
            columns=[x_name, y_name, hue_name],
        ),
    )
    _plot_fig(out_dir, f"{filename}-{y_name}")


def _plot_n_components(out_dir, filename, model_names, n_components):
    lp_data, knn_data = [], []
    for n_c in tqdm.tqdm(n_components):
        for model_name in model_names:
            lp, knn, tsne = _get_metrics(out_dir, f"{model_name}-{n_c}.npy")
            lp_data.extend([[str(n_c), _, model_name] for _ in lp])
            knn_data.extend([[str(n_c), _, model_name] for _ in knn])
            _plot_tsne(out_dir, f"{model_name}-{n_c}", tsne)
    _plot_metric(out_dir, filename, lp_data, "n_components", "lp", "model")
    _plot_metric(out_dir, filename, knn_data, "n_components", "knn", "model")


def _plot_param(out_dir, filename, param_name, params):
    data = []
    for param in tqdm.tqdm(params):
        lp, knn, tsne = _get_metrics(out_dir, f"{filename}{param}.npy")
        data.extend([[str(param), _, "lp"] for _ in lp])
        data.extend([[str(param), _, "knn"] for _ in knn])
        _plot_tsne(out_dir, f"{filename}{param}", tsne)
    _plot_metric(out_dir, filename, data, param_name, "acc", "metric")


def plot_models(out_dir, model_names, n_components, noise_stds, mask_probs, betas):
    n_components and _plot_n_components(out_dir, "NC", model_names, n_components)
    noise_stds and _plot_param(out_dir, "DAE-128-noise", "noise_std", noise_stds)
    mask_probs and _plot_param(out_dir, "MAE-128-mask", "mask_prob", mask_probs)
    betas and _plot_param(out_dir, "VAE-128-beta", "beta", betas)


if __name__ == "__main__":
    out_dir, model_names, n_components, noise_stds, mask_probs, betas = (
        "./out",
        ["RP", "PCA", "AE", "DAE", "MAE", "VAE", "SimCLR"],
        [128, 256, 512, 1024],
        [0, 0.1, 0.25, 0.5, 1],
        [0, 0.1, 0.25, 0.5, 0.75],
        [1e-4, 1e-3, 1e-2, 1e-1],
    )
    run_models(out_dir, model_names, n_components, noise_stds, mask_probs, betas)
    plot_models(out_dir, model_names, n_components, noise_stds, mask_probs, betas)
