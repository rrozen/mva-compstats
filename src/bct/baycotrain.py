"""
Gaussian Processes
==================

Underlying function f. Observations are Y = f(X) + epsilon, with epsilon gaussian noise with parameter sigma.

- Observe Y from X
- Predict f(X')


// Without noise taken into account

    [f(X), f(X')] ~ Normal(0, K([X, X'], [X, X']))

Hence f(X') | f(X) ~ Normal(mu(X'), cov(X')) where
    mu(X') = K(X', X) • inv(K(X, X)) • Y
    cov(X') = ...
Prediction :

    f' = K(X', X) • inv(K(X, X)) • Y

// With noise taken into account

    f' = K(X', X) • inv(K(X, X) + sigma^2•I) • Y
"""

import numpy as np
from sklearn.gaussian_process.kernels import Kernel, RBF  # type: ignore


def getKc(X, sigs=None, rho=1 / np.sqrt(2), missing_value=None):
    n, d = X.shape
    if sigs is None:
        sigs = [1e-3] * d

    if missing_value is not None:
        X[X == missing_value] = np.nan

    if np.isnan(X).any():  # missing views
        Lambda = np.zeros((n, n))
        for i, sig in enumerate(sigs):
            view = np.arange(n)[~np.isnan(X[:, i])]
            xx, yy = np.meshgrid(view, view)
            Lambda[xx, yy] += np.linalg.inv(
                RBF(rho)(X[view, i, None]) + sig ** 2 * np.identity(view.size)
            )
        return np.linalg.inv(Lambda)

    ks = [RBF(rho)(X[:, i, None]) for i in range(d)]

    return np.linalg.inv(
        sum([np.linalg.inv(k + (sig ** 2) * np.identity(n)) for (k, sig) in zip(ks, sigs)])
    )


class Consensus(Kernel):
    def __init__(self, sigs=None, rho=1, missing_value=None):
        self.rho = rho
        self.sigs = sigs
        self.missing_value = missing_value

    @property
    def requires_vector_input(self):
        return False

    def __call__(self, X, Y=None, eval_gradient=False):
        if Y is None:
            return getKc(X, self.sigs, self.rho, missing_value=self.missing_value)
        else:
            ntrain, _ = X.shape
            Kc = getKc(
                np.concatenate((X, Y)), self.sigs, self.rho, missing_value=self.missing_value
            )
            return Kc[:ntrain, ntrain:]

    def is_stationary(self):
        return False

    def diag(self, X):
        pass
