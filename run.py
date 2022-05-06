"""This module contains functions to run experiments and plot results."""

import os
import pickle
from collections import defaultdict
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tqdm

import trainer

sns.color_palette("deep")
sns.set_theme(style="darkgrid")


def _run_trials(
    out_dir: str,
    filename: str,
    n_trials: int = 5,
    **kwargs,
):
    results = defaultdict(list)
    for _ in range(n_trials):
        lp, knn, tsne = trainer.train(**kwargs)
        results["lp"].append(lp)
        results["knn"].append(knn)
        results["tsne"].append(tsne)
        print(f"experiment: {filename} lp-acc: {lp} knn-acc: {knn}")
    with open(os.path.join(out_dir, f"{filename}.pkl"), "wb+") as f:
        pickle.dump(results, f)


def _run_models(
    out_dir: str,
    model_names: Tuple[str],
    n_components: Tuple[int],
    sigmas: Tuple[float],
    betas: Tuple[float],
):
    for model_name in model_names:
        for nc in n_components:
            _run_trials(
                out_dir,
                f"{model_name}-{nc}",
                model_name=model_name,
                n_components=nc,
            )
    for sigma in sigmas:
        _run_trials(
            out_dir,
            f"dae-128-sigma{sigma}",
            model_name="dae",
            n_components=128,
            sigma=sigma,
        )
    for beta in betas:
        _run_trials(
            out_dir,
            f"vae-128-beta{beta}",
            model_name="vae",
            n_components=128,
            beta=beta,
        )


def _get_metrics(
    out_dir: str, filename: str
) -> Tuple[List[float], List[float], np.ndarray]:
    with open(os.path.join(out_dir, f"{filename}.pkl"), "rb") as f:
        data = pickle.load(f)
        lp = data.get("lp")
        knn = data.get("knn")
        tsne = data.get("tsne")[0]
        return lp, knn, tsne


def _plot_fig(
    out_dir: str,
    filename: str,
    show_legend: bool = True,
):
    if show_legend:
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
    out_dir: str,
    filename: str,
    tsne: np.ndarray,
    class_names: Tuple[str] = (
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
    ),
):
    ax = sns.scatterplot(
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
    ax.set(
        xlabel=None,
        ylabel=None,
        xticklabels=[],
        yticklabels=[],
    )
    _plot_fig(out_dir, f"{filename}-tsne")


def _plot_metric(
    out_dir: str,
    filename: str,
    data: List[Tuple[Union[str, float]]],
    x_name: str,
    y_name: str,
    hue_name: Optional[str] = None,
):
    sns.lineplot(
        x=x_name,
        y=y_name,
        hue=hue_name,
        ci=95,
        data=pd.DataFrame(
            data,
            columns=((x_name, y_name, hue_name) if hue_name else (x_name, y_name)),
        ),
    )
    _plot_fig(out_dir, f"{filename}-{y_name}", show_legend=bool(hue_name))


def _plot_n_components(
    out_dir: str,
    filename: str,
    model_names: Tuple[str],
    n_components: Tuple[int],
):
    lp_data, knn_data = [], []
    for nc in tqdm.tqdm(n_components):
        for model_name in model_names:
            lp, knn, tsne = _get_metrics(out_dir, f"{model_name}-{nc}")
            lp_data.extend([(str(nc), _, model_name) for _ in lp])
            knn_data.extend([(str(nc), _, model_name) for _ in knn])
            _plot_tsne(out_dir, f"{model_name}-{nc}", tsne)
    _plot_metric(out_dir, filename, lp_data, "n_components", "lp", "model")
    _plot_metric(out_dir, filename, knn_data, "n_components", "knn", "model")


def _plot_param(
    out_dir: str,
    filename: str,
    param_name: str,
    params: Tuple[float],
):
    lp_data, knn_data = [], []
    for param in tqdm.tqdm(params):
        lp, knn, tsne = _get_metrics(out_dir, f"{filename}{param}")
        lp_data.extend([(str(param), _) for _ in lp])
        knn_data.extend([(str(param), _) for _ in knn])
        _plot_tsne(out_dir, f"{filename}{param}", tsne)
    _plot_metric(out_dir, filename, lp_data, param_name, "lp")
    _plot_metric(out_dir, filename, knn_data, param_name, "knn")


def _plot_models(
    out_dir: str,
    model_names: Tuple[str],
    n_components: Tuple[int],
    sigmas: Tuple[float],
    betas: Tuple[float],
):
    _plot_n_components(out_dir, "nc", model_names, n_components)
    _plot_param(out_dir, "dae-128-sigma", "sigma", sigmas)
    _plot_param(out_dir, "vae-128-beta", "beta", betas)


def main(
    out_dir: str = "./out",
    model_names: Tuple[str] = ("rp", "pca", "ae", "dae", "vae", "simclr"),
    n_components: Tuple[int] = (128, 256, 512, 1024),
    sigmas: Tuple[float] = (0.1, 0.5, 0.75, 1, 1.5),
    betas: Tuple[float] = (1e-4, 1e-3, 1e-2, 1e-1),
):
    """Runs experiments and plot results.

    Args:
        out_dir:
            Path to output experiment results.
        model_names:
            Models to perform experiments on. Supports Random Projection (rp),
            Principle Component Analysis (pca), Autoencoder (ae),
            Denosing Autoencoder (dae), Variantional Autoencoder (vae) and
            Contrastive Learning (simclr).
        n_components:
            Dimensionality reduction feature dimensions.
        sigmas:
            Noise standard deviations for Denosing Autoencoder experiments.
        betas:
            Beta values for Variantional Autoencoder experiments.
    """
    _run_models(out_dir, model_names, n_components, sigmas, betas)
    _plot_models(out_dir, model_names, n_components, sigmas, betas)


if __name__ == "__main__":
    main()
