"""
Gaussian Processes
==================

Underlying function f. Observations are Y = f(X) + epsilon, with epsilon gaussian noise with parameter sigma.

- Observe Y from X
- Predict f(X')


// Without noise taken into account

    [f(X), f(X')] ~ Normal(0, K([X, X'], [X, X']))

Hence f(X') | f(X) ~ Normal(mu(X'), cov(X')) where
    mu(X') = K(X', X) . inv(K(X, X)) . Y
    cov(X') = ...
Prediction :

    f' = K(X', X) . inv(K(X, X)) . Y

// With noise taken into account

    f' = K(X', X) . inv(K(X, X) + sigma^2.I) . Y
"""

import numpy as np
from typing import Tuple, List, Optional
from sklearn.gaussian_process.kernels import Kernel, RBF, Hyperparameter  # type: ignore
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier  # type: ignore


def computeKc(
    X: np.ndarray,
    sig2s: Optional[List[float]] = None,
    rho: float = 1 / np.sqrt(2),
    missing_value: Optional[float] = None,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    X_ = X.copy()
    n, d = X_.shape
    if sig2s is None:
        sig2s = [1] * d

    if missing_value is not None:
        X_[X_ == missing_value] = np.nan

    if np.isnan(X_).any():  # missing views
        invs = []
        for i, sig2 in enumerate(sig2s):
            A = np.zeros((n, n))
            view = np.arange(n)[~np.isnan(X_[:, i])]
            xx, yy = np.meshgrid(view, view)
            A[xx, yy] = np.linalg.inv(RBF(rho)(X_[view, i, None]) + sig2 * np.identity(view.size))
            invs.append(A)
    else:
        ks = [RBF(rho)(X_[:, i, None]) for i in range(d)]
        invs = [np.linalg.inv(k + sig2 * np.identity(n)) for (k, sig2) in zip(ks, sig2s)]

    return np.linalg.inv(sum(invs)), invs


class Consensus(Kernel):
    def __init__(self, sig2s, sig2s_bounds, rho=1, missing_value=None):
        self.rho = rho
        self.sig2s = sig2s
        self.sig2s_bounds = sig2s_bounds
        self.missing_value = missing_value

    @property
    def hyperparameter_length_scale(self):
        return Hyperparameter("sig2s", "numeric", self.sig2s_bounds, len(self.sig2s))

    @property
    def requires_vector_input(self):
        return False

    def __call__(self, X, Y=None, eval_gradient=False):
        if Y is None:
            Kc, invs = computeKc(X, self.sig2s, self.rho, missing_value=self.missing_value)
        else:
            if eval_gradient:
                raise ValueError("Gradient can only be evaluated when Y is None.")
            ntrain, _ = X.shape
            Kc, invs = computeKc(
                np.concatenate((X, Y)), self.sig2s, self.rho, missing_value=self.missing_value
            )
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
        return np.diag(self.__call__(X))


class MyGPRegressor(GaussianProcessRegressor):
    def predict(self, X, return_std=False, return_cov=False):
        ntrain, _ = self.X_train_.shape
        K = self.kernel_(np.concatenate((self.X_train_, X)))[:ntrain, :ntrain]
        self.alpha_ = np.linalg.solve(K, self.y_train_)
        return super().predict(X, return_std, return_cov)


class MyGPClassifier(GaussianProcessClassifier):
    def update_cache(self, X):
        ntrain, _ = self.base_estimator_.X_train_.shape
        K = self.kernel_(np.concatenate((self.base_estimator_.X_train_, X)))[:ntrain, :ntrain]
        _, (self.base_estimator_.pi_, self.base_estimator_.W_sr_, self.base_estimator_.L_, _, _
            ) = self.base_estimator_._posterior_mode(
            K, return_temporaries=True
        )

    def predict(self, X):
        self.update_cache(X)
        return super().predict(X)

    def predict_proba(self, X):
        self.update_cache(X)
        return super().predict_proba(X)
