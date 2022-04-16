from sklearnex.linear_model import LogisticRegression
from sklearnex.neighbors import KNeighborsClassifier


def compute_lp(Z_tr, Y_tr, Z_te, Y_te):
    lp = LogisticRegression(random_state=0, max_iter=1000).fit(Z_tr, Y_tr)
    acc = lp.score(Z_te, Y_te)
    print(f"ACC:{acc:.3f}")


def compute_knn(Z_tr, Y_tr, Z_te, Y_te):
    knn = KNeighborsClassifier(n_neighbors=1, algorithm="ball_tree").fit(Z_tr, Y_tr)
    acc = knn.score(Z_te, Y_te)
    print(f"ACC:{acc:.3f}")
