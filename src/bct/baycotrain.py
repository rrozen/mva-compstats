import numpy as np
from typing import Tuple, List
from sklearn.gaussian_process.kernels import Kernel, RBF, Hyperparameter  # type: ignore


def compute_Kc(
    x: np.ndarray, sig2s=None, rho: float = 1 / np.sqrt(2)
) -> Tuple[np.ndarray, List[np.ndarray]]:
    n, d = x.shape
    ks = [RBF(rho)(x[:, i, None]) for i in range(d)]
    if sig2s is None:
        sig2s = [1] * d
    invs = [np.linalg.inv(k + sig2 * np.identity(n)) for (k, sig2) in zip(ks, sig2s)]
    return np.linalg.inv(sum(invs)), invs


class Consensus(Kernel):
    def __init__(self, sig2s, sig2s_bounds, rho=1):
        self.rho = rho
        self.sig2s = sig2s
        self.sig2s_bounds = sig2s_bounds

    @property
    def hyperparameter_length_scale(self):
        return Hyperparameter("sig2s", "numeric", self.sig2s_bounds, len(self.sig2s))

    def __call__(self, X, Y=None, eval_gradient=False):
        if Y is None:
            Kc, invs = compute_Kc(X, self.sig2s, self.rho)
        else:
            if eval_gradient:
                raise ValueError("Gradient can only be evaluated when Y is None.")
            ntrain, _ = X.shape
            Kc, invs = compute_Kc(np.concatenate((X, Y)), self.sig2s, self.rho)
            Kc = Kc[:ntrain, ntrain:]

        if eval_gradient:
            Kc_gradient = np.stack(
                [Kc @ inv @ inv @ Kc / sig2 for inv, sig2 in zip(invs, self.sig2s)], axis=-1
            )
            return Kc, Kc_gradient
        else:
            return Kc

    def is_stationary(self):
        return False

    def diag(self, X):
        pass
