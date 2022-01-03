import numpy as np
import scipy.stats as stats  # type: ignore
from scipy.special import logsumexp  # type: ignore
from typing import Tuple, List, Optional, Union

EPSILON = 1e-10


class Density:
    # Blueprint for all densities

    @property
    def dim(self) -> int:
        raise NotImplementedError(
            f"Property dim for density {type(self).__name__} is not implemented."
        )

    def logpdf(self, x: np.ndarray) -> Union[float, np.ndarray]:
        raise NotImplementedError(
            f"Method logpdf for density {type(self).__name__} is not implemented."
        )

    def pdf(self, x: np.ndarray) -> Union[float, np.ndarray]:
        return np.exp(self.logpdf(x))

    def sample(self, size: Union[int, Tuple[int, ...]] = 1):
        raise NotImplementedError(
            f"Method logpdf for density {type(self).__name__} is not implemented."
        )


class Gaussian(Density):
    def __init__(self, mu: Union[float, np.ndarray] = 0.0, sigma: Union[float, np.ndarray] = 1):
        mu, sigma = np.array(mu), np.array(sigma)
        assert mu.ndim <= 1
        self._dim = mu.size

        if sigma.size == 1:
            sigma = np.diag(np.diag(np.ones(self._dim) * sigma)).squeeze()
        self.normal = (
            stats.norm(loc=mu, scale=sigma)
            if self._dim == 1
            else stats.multivariate_normal(mean=mu, cov=sigma)
        )

    @property
    def dim(self):
        return self._dim

    def logpdf(self, x: np.ndarray) -> Union[float, np.ndarray]:
        assert x.shape[-1] == self.dim
        return self.normal.logpdf(x)

    def sample(self, size: Union[int, Tuple[int, ...]] = 1) -> np.ndarray:
        return self.normal.rvs(size=size)


class Mixture(Density):
    "Combines densities into a mixture."

    def __init__(self, ds: List[Density],
                 clusters_map: Optional[List[int]] = None,
                 p: Optional[List[float]] = None):
        self._dim = ds[0].dim
        assert all(d.dim == self._dim for d in ds)

        self.m = len(ds)  # number of gaussians
        if clusters_map is None:
            clusters_map = np.arange(self.m)
        self.clusters_map = clusters_map
        if p is None:
            p = np.ones(self.m) / self.m
        self.p = np.array(p)
        assert self.p.sum() - 1 < EPSILON

        self.ds = ds

    @property
    def dim(self):
        return self._dim

    def logpdf(self, x: np.ndarray) -> Union[float, np.ndarray]:
        assert x.shape[-1] == self.dim
        return logsumexp([d.logpdf(x) for d in self.ds], b=self.p[:, None], axis=0)

    def sample(self, size: Union[int, Tuple[int, ...]] = 1) -> Tuple[np.ndarray, np.ndarray]:
        if isinstance(size, int):
            size = (size,)
        gaussians = np.random.randint(self.m, size=size)
        ret_size = *size, self.dim
        ret = np.zeros(ret_size)
        clusters = np.zeros_like(gaussians)
        for k in range(self.m):
            ret[gaussians == k, :] = self.ds[k].sample(size=np.sum(gaussians == k))
            clusters[gaussians == k] = self.clusters_map[k]
        return ret.squeeze(), clusters
