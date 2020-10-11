import numpy as np
import scipy.linalg as la


def fit_ls_soft(x, y, C):
    """
    Fit alphas for dual problem soft margin Least Squares SVM.

    :param numpy.array x: instances.
    :param numpy.array y: labels.
    :param float C: penalty.
    :return: alphas
    """
    N = x.shape[0]
    # obtain the kernel
    omega = np.dot(x, x.T)
    y_omega = np.dot(y.T, omega)
    y_omega_y = np.dot(y_omega, y)
    # Build the first row of the matrix
    top_row = np.concatenate(np.zeros((1, 1), y.T))
    # Build the last N + 1 rows of the matrix
    quadrant = np.eye(N) / C + y_omega_y
    bottom_row = np.concatenate(y, quadrant)
    # Build the matrix
    A = np.concatenate(top_row, bottom_row)
    # Build the objective matrix
    B = np.concatenate(np.zeros((1, 1)), np.ones(N)).T
    # Solve the matrix problem
    solution = inverse_solver(A, B)
    b = solution[0]
    alphas = solution[1:]
    return b, alphas


def inverse_solver(a, b):
    """
    Inverse of a symmetric matrix.

    :param a:
    :param b:
    :return:
    """
    x_sp_solve = la.solve(a=a,
                          b=b,
                          lower=False,
                          overwrite_a=True,
                          overwrite_b=True,
                          debug=None,
                          check_finite=False,
                          transposed=False,
                          assume_a='sym')
    return x_sp_solve