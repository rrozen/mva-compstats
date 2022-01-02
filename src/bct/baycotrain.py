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


def getKc(x, sigs=None, rho=1 / np.sqrt(2)):
    n, d = x.shape
    ks = [RBF(rho)(x[:, i, None]) for i in range(d)]
    if sigs is None:
        sigs = [1e-3] * d
    return np.linalg.inv(
        sum([np.linalg.inv(k + (sig ** 2) * np.identity(n)) for (k, sig) in zip(ks, sigs)])
    )


class Consensus(Kernel):
    def __init__(self, sigs, rho=1):
        self.rho = rho
        self.sigs = sigs

    def __call__(self, X, Y=None, eval_gradient=False):
        if Y is None:
            return getKc(X, self.sigs, self.rho)
        else:
            ntrain, _ = X.shape
            Kc = getKc(np.concatenate((X, Y)), self.sigs, self.rho)
            return Kc[:ntrain, ntrain:]

    def is_stationary(self):
        return False

    def diag(self, X):
        pass
