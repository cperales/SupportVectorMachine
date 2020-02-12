import numpy as np
from cvxopt import matrix, solvers

solvers.options['show_progress'] = False


def fit(x, y):
    """
    Fit alphas for dual problem hard margin SVM.

    :param x: instances.
    :param y: labels.
    :return: alphas
    """
    num = x.shape[0]
    dim = x.shape[1]
    # we'll solve the dual
    # obtain the kernel
    K = y[:, None] * x
    K = np.dot(K, K.T)
    # Same that np.dot(np.dot(np.eye(num), K), K.T)
    P = matrix(K)
    q = matrix(-np.ones((num, 1)))
    G = matrix(-np.eye(num))
    h = matrix(np.zeros(num))
    A = matrix(y.reshape(1, -1))
    b = matrix(np.zeros(1))
    solvers.options['show_progress'] = False
    sol = solvers.qp(P, q, G, h, A, b)
    alphas = np.array(sol['x'])
    return alphas


def fit_soft(x, y, C):
    """
    Fit alphas for dual problem soft margin SVM.

    :param numpy.array x: instances.
    :param numpy.array y: labels.
    :param float C: penalty.
    :return: alphas
    """
    num = x.shape[0]
    dim = x.shape[1]
    # we'll solve the dual
    # obtain the kernel
    K = y[:, None] * x
    K = np.dot(K, K.T)
    P = matrix(K)
    q = matrix(-np.ones((num, 1)))
    g = np.concatenate((-np.eye(num), np.eye(num)))
    G = matrix(g)
    h_array = np.concatenate((np.zeros(num), C * np.ones(num)))
    h = matrix(h_array)
    # G = matrix(np.eye(num))
    # h = matrix(np.ones(num))
    A = matrix(y.reshape(1, -1))
    b = matrix(np.zeros(1))
    sol = solvers.qp(P, q, G, h, A, b)
    alphas = np.array(sol['x'])
    return alphas


def fit_kernel(x, y, C, kernel_fun):
    """
    Fit alphas for dual problem soft margin Kernel SVM.

    :param numpy.array x: instances.
    :param numpy.array y: labels.
    :param float C: penalty.
    :param kernel_fun: kernel function.
    :return: alphas
    """
    num = x.shape[0]
    dim = x.shape[1]
    # we'll solve the dual
    # obtain the kernel
    y_vec = y.reshape((num, 1))
    y_matrix = np.dot(y_vec, y_vec.T)
    k = kernel_fun(x)
    K = y_matrix * k
    P = matrix(K)
    q = matrix(-np.ones((num, 1)))
    g = np.concatenate((-np.eye(num), np.eye(num)))
    G = matrix(g)
    h_array = np.concatenate((np.zeros(num), C * np.ones(num)))
    h = matrix(h_array)
    # G = matrix(np.eye(num))
    # h = matrix(np.ones(num))
    A = matrix(y.reshape(1, -1))
    b = matrix(np.zeros(1))
    sol = solvers.qp(P, q, G, h, A, b)
    alphas = np.array(sol['x'])
    return alphas
