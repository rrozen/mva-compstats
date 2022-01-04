import matplotlib.pyplot as plt  # type: ignore
import numpy as np


def plot_pdf(pdf, bound=1):
    if np.isscalar(bound):
        tx = np.linspace(-bound, bound, 200)
        ty = np.linspace(-bound, bound, 200)
    else:
        mini, maxi = bound
        if np.isscalar(mini) and np.isscalar(maxi):
            tx = np.linspace(mini, maxi, 200)
            ty = np.linspace(mini, maxi, 200)
        else:
            tx = np.linspace(mini[0], mini[1], 200)
            ty = np.linspace(maxi[0], maxi[1], 200)
    xx, yy = np.meshgrid(tx, ty)
    xy = np.stack([xx.flatten(), yy.flatten()], axis=1)
    plt.scatter(xy[:, 0], xy[:, 1], c=pdf(xy), cmap="GnBu", alpha=0.5)


def mask_views(X, ps):
    ps = np.cumsum(np.array(ps))
    assert ps[-1] <= 1
    x = X.copy()
    n, d = x.shape
    for i in range(n):
        u = np.random.random()
        idx = np.searchsorted(ps, u)
        if idx < d:
            x[i, idx] = 0
    return x
