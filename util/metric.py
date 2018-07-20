import numpy as np


def accuracy(clf, X, y, y_pred=None):
    """

    :param clf:
    :param X:
    :param y:
    :param y_pred:
    :return: accuracy
    """
    if y_pred is None:
        y_pred = clf.predict(X=X)
    comp = np.array((y_pred == y), dtype=np.float)
    return np.mean(comp)
