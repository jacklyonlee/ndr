import os

import numpy as np

import trainer


def _run_trials(out_dir, filename, n_trials=10, **kwargs):
    results = {"LP": [], "KNN": [], "TSNE": []}
    for _ in range(n_trials):
        lp, knn, tsne = trainer.train(**kwargs)
        print(filename, "LP", lp, "KNN", knn)
        results["LP"].append(lp)
        results["KNN"].append(knn)
        results["TSNE"].append(tsne)
    with open(os.path.join(out_dir, filename), "wb+") as f:
        np.save(f, results)


def run(out_dir):
    for model_name in ["RP", "PCA", "AE", "DAE", "MAE"]:
        for n_components in [128, 256, 512]:
            if model_name == "DAE":
                for noise_std in [0, 0.1, 0.25, 0.5, 1]:
                    _run_trials(
                        out_dir,
                        f"{model_name}-{noise_std}",
                        model_name=model_name,
                        n_components=n_components,
                        noise_std=noise_std,
                    )
            elif model_name == "MAE":
                for mask_prob in [0, 0.1, 0.25, 0.5, 0.75, 1]:
                    _run_trials(
                        out_dir,
                        f"{model_name}-{mask_prob}",
                        model_name=model_name,
                        n_components=n_components,
                        mask_prob=mask_prob,
                    )
            else:
                _run_trials(
                    out_dir,
                    model_name,
                    model_name=model_name,
                    n_components=n_components,
                )


if __name__ == "__main__":
    run("./out")
