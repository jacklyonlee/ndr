import numpy as np
from sklearnex.linear_model import LogisticRegression
from sklearnex.manifold import TSNE
from sklearnex.neighbors import KNeighborsClassifier


def compute_lp(
    Z_tr: np.ndarray,
    Y_tr: np.ndarray,
    Z_te: np.ndarray,
    Y_te: np.ndarray,
):
    lp = LogisticRegression(random_state=0, max_iter=5000).fit(Z_tr, Y_tr)
    return lp.score(Z_te, Y_te)


def compute_knn(
    Z_tr: np.ndarray,
    Y_tr: np.ndarray,
    Z_te: np.ndarray,
    Y_te: np.ndarray,
):
    knn = KNeighborsClassifier(n_neighbors=1, algorithm="ball_tree").fit(Z_tr, Y_tr)
    return knn.score(Z_te, Y_te)


def compute_tsne(Z: np.ndarray, Y: np.ndarray):
    emb = TSNE(n_components=2, learning_rate="auto", init="pca").fit_transform(Z)
    return np.concatenate((emb, Y[:, np.newaxis]), axis=1)
