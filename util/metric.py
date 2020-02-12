import numpy as np


def accuracy(clf, X, y):
    """

    :param clf:
    :param X:
    :param y:
    :param y_pred:
    :return: accuracy
    """
    y_pred = clf.predict(X=X)
    comp = np.array((y_pred == y), dtype=np.float)
    acc = np.mean(comp)
    return np.mean(comp)
