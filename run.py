import os
import pickle
from typing import List, Tuple, Union

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
) -> None:
    results = {"lp": [], "knn": [], "tsne": []}
    for _ in range(n_trials):
        lp, knn, tsne = trainer.train(**kwargs)
        print(filename, "lp-acc", lp, "knn-acc", knn)
        results["lp"].append(lp)
        results["knn"].append(knn)
        results["tsne"].append(tsne)
    with open(os.path.join(out_dir, f"{filename}.pkl"), "wb+") as f:
        pickle.dump(results, f)


def _run_models(
    out_dir: str,
    model_names: List[str],
    n_components: List[int],
    noise_stds: List[float],
    betas: List[float],
) -> None:
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
            f"dae-128-noise{noise_std}",
            model_name="dae",
            n_components=512,
            noise_std=noise_std,
        )
    # test different betas
    for beta in betas:
        _run_trials(
            out_dir,
            f"vae-128-beta{beta}",
            model_name="vae",
            n_components=512,
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


def _plot_fig(out_dir: str, filename: str) -> None:
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
    class_names: List[str] = [
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
) -> None:
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
    data: List[List[Union[str, float]]],
    x_name: str,
    y_name: str,
    hue_name: str,
) -> None:
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


def _plot_n_components(
    out_dir: str, filename: str, model_names: List[str], n_components: List[int]
) -> None:
    lp_data, knn_data = [], []
    for nc in tqdm.tqdm(n_components):
        for model_name in model_names:
            lp, knn, tsne = _get_metrics(out_dir, f"{model_name}-{nc}")
            lp_data.extend([[str(nc), _, model_name] for _ in lp])
            knn_data.extend([[str(nc), _, model_name] for _ in knn])
            _plot_tsne(out_dir, f"{model_name}-{nc}", tsne)
    _plot_metric(out_dir, filename, lp_data, "n_components", "lp", "model")
    _plot_metric(out_dir, filename, knn_data, "n_components", "knn", "model")


def _plot_param(
    out_dir: str, filename: str, param_name: str, params: List[float]
) -> None:
    data = []
    for param in tqdm.tqdm(params):
        lp, knn, tsne = _get_metrics(out_dir, f"{filename}{param}")
        data.extend([[str(param), _, "lp"] for _ in lp])
        data.extend([[str(param), _, "knn"] for _ in knn])
        _plot_tsne(out_dir, f"{filename}{param}", tsne)
    _plot_metric(out_dir, filename, data, param_name, "acc", "metric")


def _plot_models(
    out_dir: str,
    model_names: List[str],
    n_components: List[int],
    noise_stds: List[float],
    betas: List[float],
) -> None:
    n_components and _plot_n_components(out_dir, "nc", model_names, n_components)
    noise_stds and _plot_param(out_dir, "dae-128-noise", "noise_std", noise_stds)
    betas and _plot_param(out_dir, "vae-128-beta", "beta", betas)


def main(
    out_dir: str = "./out",
    model_names: List[str] = ["rp", "pca", "ae", "dae", "vae", "simclr"],
    n_components: List[int] = [128, 256, 512, 1024],
    noise_stds: List[float] = [0.1, 0.25, 0.5, 0.75, 1],
    betas: List[float] = [1e-4, 1e-3, 1e-2, 1e-1],
) -> None:
    """Script to run experiments and plot results.

    Parameters
    ----------
    out_dir : str
        Path to output experiment results.
    model_name : List[str]
        List of models to perform n_components experiments on.
        Supports Random Projection (rp), Principle Component Analysis (pca),
        Autoencoder (ae), Denosing Autoencoder (dae),
        Variantional Autoencoder (vae) and Contrastive Learning (simclr).
    n_components : List[int]
        Dimensionality reduction output dimensions.
    noise_stds : List[float]
        Noise levels for Denosing Autoencoder experiments.
        See model.ndr for more details.
    betas : List[float]
        Beta values for Variantional Autoencoder experiments.
        See model.ndr for more details.
    """
    _run_models(out_dir, model_names, n_components, noise_stds, betas)
    _plot_models(out_dir, model_names, n_components, noise_stds, betas)


if __name__ == "__main__":
    main()
