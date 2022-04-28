import os

import numpy as np

import trainer


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


def run_models(out_dir):
    # test different n_components
    for model_name in ["RP", "PCA", "AE", "DAE", "MAE", "VAE", "SimCLR"]:
        for n_components in [128, 256, 512, 1024]:
            _run_trials(
                out_dir,
                f"{model_name}-{n_components}",
                model_name=model_name,
                n_components=n_components,
            )

    # test different noise levels
    for noise_std in [0, 0.1, 0.25, 0.5, 1]:
        _run_trials(
            out_dir,
            f"DAE-512-NS{noise_std}",
            model_name="DAE",
            n_components=512,
            noise_std=noise_std,
        )

    # test different mask probs
    for mask_prob in [0, 0.1, 0.25, 0.5, 0.75]:
        _run_trials(
            out_dir,
            f"MAE-512-MP{mask_prob}",
            model_name="MAE",
            n_components=512,
            mask_prob=mask_prob,
        )

    # test different betas
    for beta in [1e-3, 1e-2, 1e-1]:
        _run_trials(
            out_dir,
            f"VAE-512-B{beta}",
            model_name="VAE",
            n_components=512,
            beta=beta,
        )


def plot_models(out_dir):
    pass


if __name__ == "__main__":
    out_dir = "./out"
    run_models(out_dir)
    plot_models(out_dir)
