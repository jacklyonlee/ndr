import numpy as np
import sklearnex.linear_model as linear_model
import sklearnex.manifold as manifold
import sklearnex.neighbors as neighbors


def compute_lp(
    z_tr: np.ndarray,
    y_tr: np.ndarray,
    z_te: np.ndarray,
    y_te: np.ndarray,
) -> float:
    """Computes linear probe accuracy.

    Parameters
    ----------
    z_tr: np.ndarray
        Dimension reduced training data with shape (N, n_components).
    y_tr: np.ndarray
        Labels of training data with shape (N,).
    z_te: np.ndarray
        Dimension reduced test data with shape (N, n_components).
    y_te: np.ndarray
        Labels of test data with shape (N,).

    Returns
    -------
    float
        Linear probe accuracy.
    """
    lp = linear_model.LogisticRegression(
        random_state=0,
        max_iter=5000,
    ).fit(z_tr, y_tr)
    return float(lp.score(z_te, y_te))


def compute_knn(
    z_tr: np.ndarray,
    y_tr: np.ndarray,
    z_te: np.ndarray,
    y_te: np.ndarray,
) -> float:
    """Computes nearest neighbor accuracy.

    Parameters
    ----------
    z_tr: np.ndarray
        Dimension reduced training data with shape (N, n_components).
    y_tr: np.ndarray
        Labels of training data with shape (N,).
    z_te: np.ndarray
        Dimension reduced test data with shape (N, n_components).
    y_te: np.ndarray
        Labels of test data with shape (N,).

    Returns
    -------
    float
        Nearest neighbor accuracy.
    """
    knn = neighbors.KNeighborsClassifier(
        n_neighbors=1,
        algorithm="ball_tree",
    ).fit(z_tr, y_tr)
    return float(knn.score(z_te, y_te))


def compute_tsne(z: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Computes nearest neighbor accuracy.

    Parameters
    ----------
    z: np.ndarray
        Dimension reduced test data of shape (N, n_components).
    y: np.ndarray
        Labels of test data of shape (N,).

    Returns
    -------
    np.ndarray
        t-SNE embeddings concatenated with class labels with shape (N, 3).
    """
    emb = manifold.TSNE(
        n_components=2,
        learning_rate="auto",
        init="pca",
    ).fit_transform(z)
    return np.concatenate((emb, y[:, np.newaxis]), axis=1)
