import numpy as np
from sklearn.gaussian_process.kernels import Kernel, RBF  # type: ignore


def compute_Kc(x, sigs=None, rho=1 / np.sqrt(2)):
    n, d = x.shape
    ks = [RBF(rho)(x[:, i, None]) for i in range(d)]
    if sigs is None:
        sigs = [1e-3] * d
    return np.linalg.inv(
        sum([np.linalg.inv(k + (sig ** 2) * np.identity(n)) for (k, sig) in zip(ks, sigs)])
    )


class consensus(Kernel):
    def __init__(self, sigs, rho=1):
        self.rho = rho
        self.sigs = sigs

    def __call__(self, X, Y=None, eval_gradient=False):
        if Y is None:
            return compute_Kc(X, self.sigs, self.rho)
        else:
            ntrain, _ = X.shape
            Kc = compute_Kc(np.concatenate((X, Y)), self.sigs, self.rho)
            return Kc[:ntrain, ntrain:]

    def is_stationary(self):
        return False

    def diag(self, X):
        pass


def regression(x_train, y_train, x_test):
    ntrain = len(x_train)
    x = np.concatenate((x_train, x_test))
    Kc = compute_Kc(x, [1e-3, 1e3])
    K_train = Kc[:ntrain, :ntrain]
    K_test = Kc[:ntrain, ntrain:]
    y_test = K_test.T @ K_train @ y_train
    return y_test
