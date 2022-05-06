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

    Args:
        z_tr:
            Dimension reduced training data with shape (N, n_components).
        y_tr:
            Labels of training data with shape (N,).
        z_te:
            Dimension reduced test data with shape (N, n_components).
        y_te:
            Labels of test data with shape (N,).

    Returns:
        Linear Probe accuracy.
    """
    lp = LogisticRegression(random_state=0, max_iter=5000).fit(z_tr, y_tr)
    return float(lp.score(z_te, y_te))


def compute_knn(
    z_tr: np.ndarray,
    y_tr: np.ndarray,
    z_te: np.ndarray,
    y_te: np.ndarray,
) -> float:
    """Computes Nearest Neighbor classification accuracy.

    Args:
        z_tr:
            Dimension reduced training data with shape (N, n_components).
        y_tr:
            Labels of training data with shape (N,).
        z_te:
            Dimension reduced test data with shape (N, n_components).
        y_te:
            Labels of test data with shape (N,).

    Returns:
        Nearest Neighbor classification accuracy.
    """
    knn = KNeighborsClassifier(n_neighbors=1, algorithm="ball_tree").fit(z_tr, y_tr)
    return float(knn.score(z_te, y_te))


def compute_tsne(z: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Computes t-distributed Stochastic Neighbor Embeddings (t-SNE).

    Args:
        z:
            Dimension reduced test data with shape (N, n_components).
        y:
            Labels of test data with shape (N,).

    Returns:
        t-SNE embeddings concatenated with class labels with shape (N, 3).
    """
    emb = TSNE(n_components=2, learning_rate="auto", init="pca").fit_transform(z)
    return np.concatenate((emb, y[:, np.newaxis]), axis=1)
