import numpy as np
import matplotlib.pyplot as plt


def two_gaussians(x1, y1, x2, y2, ntrain, ntest):

    x1_test = np.random.multivariate_normal(np.array([x1,y1]), np.eye(2), ntest//2)
    x2_test = np.random.multivariate_normal(np.array([x2,y2]), np.eye(2), ntest//2)

    x_test = np.concatenate((x1_test, x2_test))
    y_test = np.concatenate((np.ones(ntrain//2)*(-1), np.ones(ntrain//2)*1))

    x1_train = np.random.multivariate_normal(np.array([x1,y1]), np.eye(2), ntrain//2)
    x2_train = np.random.multivariate_normal(np.array([x2,y2]), np.eye(2), ntrain//2)

    x_train = np.concatenate((x1_train, x2_train))
    y_train = np.concatenate((np.ones(ntrain//2)*(-1), np.ones(ntrain//2)*1))
    
    plt.plot(x1_test[:,0], x1_test[:, 1], '.')
    plt.plot(x2_test[:,0], x2_test[:, 1], '.')
    plt.plot(x1_train[:,0], x1_train[:, 1], '.', markersize=15)
    plt.plot(x2_train[:,0], x2_train[:, 1], '.', markersize=15)

    return x_train, y_train, x_test, y_test