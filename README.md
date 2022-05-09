# Neural Dimensionality Reduction

This repository implements classical and neural dimensionality reduction techniques including Gaussian Random Projection, PCA, Autoencoder, Denoising Autoencoder, Variational Autoencoder and SimCLR and corresponding evaluation metrics such as Linear Probe, KNN and t-SNE.
![t-SNE](https://user-images.githubusercontent.com/103356034/167145092-487b6004-d71d-4e77-9242-5db3d55edc0c.png)

## Installation

- Set up and activate conda environment.

```bash
conda env create -f environment.yml
conda activate ndr
```

- Install pre-commit hooks.

```bash
pre-commit install
```

## Quick Start

- Train and test a single model.

```bash
python train.py
```

- Run all experiments and plot results.

```bash
python run.py
```
