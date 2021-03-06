"""This module contains functions to compute evaluation metrics."""

import numpy as np
from sklearnex.linear_model import LogisticRegression
from sklearnex.manifold import TSNE
from sklearnex.neighbors import KNeighborsClassifier


def compute_lp(
    z_tr: np.ndarray,
    y_tr: np.ndarray,
    z_te: np.ndarray,
    y_te: np.ndarray,
) -> float:
    """Computes Linear Probe accuracy.

    Parameters
    ----------
    z_tr
        Dimension reduced training data with shape (N, n_components).
    y_tr
        Labels of training data with shape (N,).
    z_te
        Dimension reduced test data with shape (N, n_components).
    y_te
        Labels of test data with shape (N,).

    Returns
    -------
    float
        Linear Probe accuracy.
    """
    lp = LogisticRegression(random_state=0, max_iter=5000)
    return float(lp.fit(z_tr, y_tr).score(z_te, y_te))


def compute_knn(
    z_tr: np.ndarray,
    y_tr: np.ndarray,
    z_te: np.ndarray,
    y_te: np.ndarray,
) -> float:
    """Computes Nearest Neighbor classification accuracy.

    Parameters
    ----------
    z_tr
        Dimension reduced training data with shape (N, n_components).
    y_tr
        Labels of training data with shape (N,).
    z_te
        Dimension reduced test data with shape (N, n_components).
    y_te
        Labels of test data with shape (N,).

    Returns
    -------
    float
        Nearest Neighbor classification accuracy.
    """
    knn = KNeighborsClassifier(n_neighbors=1, algorithm="ball_tree")
    return float(knn.fit(z_tr, y_tr).score(z_te, y_te))


def compute_tsne(z: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Computes t-distributed Stochastic Neighbor Embeddings (t-SNE).

    Parameters
    ----------
    z
        Dimension reduced test data with shape (N, n_components).
    y
        Labels of test data with shape (N,).

    Returns
    -------
    np.ndarray
        t-SNE embeddings concatenated with class labels with shape (N, 3).
    """
    tsne = TSNE(n_components=2, learning_rate="auto", init="pca")
    return np.concatenate((tsne.fit_transform(z), y[:, np.newaxis]), axis=1)
