import pickle
import numpy as np
import matplotlib.pyplot as plt

DIM = 2
NUM = 50
COLORS = ['red', 'blue']

# 2-D mean of ones
M1 = np.ones((DIM,))
# # 2-D mean of threes
M2 = 2.1 * np.ones((DIM,))
# M2 = 3 * np.ones((DIM,))
# 2-D covariance of 0.3
C1 = np.diag(0.3 * np.ones((DIM,)))
# 2-D covariance of 0.2
C2 = np.diag(0.2 * np.ones((DIM,)))


def read_data(f):
    """
    Function that read a pickle.

    :param f: filename of the data.
    :return: x, y
    """
    with open(f, 'rb') as f:
        data = pickle.load(f)
    x, y = data[0], data[1]
    return x, y


def generate_gaussian(m, c, num):
    return np.random.multivariate_normal(m, c, num)


def plot_data_with_labels(x, y):
    unique = np.unique(y)
    for li in range(len(unique)):
        x_sub = x[y == unique[li]]
        plt.scatter(x_sub[:, 0], x_sub[:, 1], c = COLORS[li])


def plot_ax_data_with_labels(x, y, ax):
    unique = np.unique(y)
    for li in range(len(unique)):
        x_sub = x[y == unique[li]]
        ax.scatter(x_sub[:, 0], x_sub[:, 1], c=COLORS[li])


def plot_separator(ax, w, b, axis_X, axis_Y, color='k'):
    slope = -w[0] / w[1]
    intercept = -b / w[1]
    min_axis_X, max_axis_X = axis_X.min(), axis_X.max()
    min_axis_Y, max_axis_Y = axis_Y.min(), axis_Y.max()
    x = np.arange(min_axis_X, max_axis_X)
    ax.plot(x, x * slope + intercept, color + '-')


def generate_data(m_1=1.0,
                  m_2=1.5,
                  c_1=0.3,
                  c_2=0.2,
                  num=50,
                  dim=2,
                  dataname=None):
    """
    Generate random binary data and save it into a pickle.
    """
    m_1 = m_1 * np.ones((dim,))
    m_2 = m_2 * np.ones((dim,))
    c_1 = np.diag(c_1 * np.ones((dim,)))
    c_2 = np.diag(c_2 * np.ones((dim,)))

    # generate points for class 1
    x1 = generate_gaussian(m_1, c_1, num)
    # generate points for class 2
    x2 = generate_gaussian(m_2, c_2, num)
    # labels
    y1 = np.ones((x1.shape[0],))
    y2 = -np.ones((x2.shape[0],))
    # join
    x = np.concatenate((x1, x2), axis=0)
    y = np.concatenate((y1, y2), axis=0)
    if dataname is not None:
        with open(dataname, 'wb') as f:
            pickle.dump((x, y), f)

    return x, y


def train_test_split(X, y, prop=0.25):
    """

    :param X:
    :param y:
    :param prop:
    :return:
    """
    index = np.random.choice(a=[False, True],
                             size=(X.shape[0],),
                             p=[prop, 1 - prop])
    X_train = X[index]
    y_train = y[index]
    X_test = X[np.invert(index)]
    y_test = y[np.invert(index)]

    return X_train, y_train, X_test, y_test


def plot_data_separator(X, y, w, b, figname='svm.png'):
    """
    Plot the data and the vector.
    """
    fig, ax = plt.subplots()
    plot_ax_data_with_labels(X, y, ax)
    axis_X = X[:, 0]
    axis_Y = X[:, 1]
    # yx_sub[:, 0], yx_sub[:, 1]
    plot_separator(ax, w, b, axis_X, axis_Y)
    plt.savefig(figname)
    plt.show()
    return fig, ax
