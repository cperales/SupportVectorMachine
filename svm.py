import logging
from svm_solver import *
from generate_data import *
logging.basicConfig(level=logging.DEBUG)


class LinearSVM(object):
    """
    Implementation of the linear support vector machine.
    """
    def __init__(self):
        self.x = None
        self.y = None
        self.w = None
        self.b = None

    def fit(self, X, y, soft=True):
        """
        Fit the model according to the given training data.

        :param numpy.array X:
        :param numpy.array y:
        :param bool soft:
        """
        # Data can be added as a pickle using read_data method
        if soft is True:
            alphas = fit_soft(X, y)
        else:
            alphas = fit(X, y)

        # get weights
        w = np.sum(alphas * y[:, None] * X, axis=0)
        # # get b
        # cond = (alphas > 1e-4).reshape(-1)
        # b = self.y[cond] - np.dot(self.x[cond], w)
        b_vector = y - np.dot(X, w)
        b = b_vector.sum() / b_vector.size

        # normalize
        norm = np.linalg.norm(w)
        w, b = w / norm, b / norm

        # Store values
        self.w = w
        self.b = b


if __name__ == '__main__':
    try:
        import os
        os.remove('images/svm.png')
        logging.debug('Previous image was removed')
    except:
        logging.debug('Previous image did not exit')

    clf = LinearSVM()
    X, y = generate_data(dataname='data/gaussiandata.pickle')
    # X,y = read_data('data/gaussiandata.pickle')
    clf.fit(X=X, y=y, soft=True)
    plot_data_separator(X,y, clf.w, clf.b, 'images/svm.png')
