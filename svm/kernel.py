import logging
from .solver import *
from .linear import LinearSVM
from util.generate_data import *
from util import kernel_dict
from functools import partial
logging.basicConfig(level=logging.DEBUG)


class KernelSVM(LinearSVM):
    """
    Implementation of the kernel version of
    Support Vector Machine (SVM).
    """
    def __init__(self):
        self.X = None
        self.y = None
        self.alphas = None
        self.b = None
        self.kernel = None

    def fit(self, X, y, kernel_type='rbf', k=1.0):
        """
        Fit the model according to the given training data.

        :param numpy.array X:
        :param numpy.array y:
        :param str kernel_type:
        :param float k:
        """
        self.kernel = partial(kernel_dict[kernel_type.lower()], k=k)
        # Get alphas
        alphas = fit_soft(X, y)
        # normalize
        alphas = alphas / np.linalg.norm(alphas)
        # Get b
        b_vector = y - np.sum(self.kernel(X) * alphas * y[:, None], axis=0)
        b = b_vector.sum() / b_vector.size

        # Store values
        self.X = X
        self.y = y
        self.alphas = alphas
        self.b = b

    def predict(self, X):
        """

        :param x:
        :return:
        """
        prod = np.sum(self.kernel(self.X, X) * self.alphas * self.y[:, None],
                      axis=0) + self.b
        y = np.sign(prod)
        return y
