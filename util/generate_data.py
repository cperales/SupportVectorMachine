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


def plot_separator(ax, w, b, color='k'):
    slope = -w[0] / w[1]
    intercept = -b / w[1]
    x = np.arange(0, 6)
    ax.plot(x, x * slope + intercept, color + '-')


def generate_data(m_1=1.0,
                  m_2=2.1,
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


def plot_data_separator(X, y, w, b, figname='svm.png'):
    """
    Plot the data and the vector.
    """
    fig, ax = plt.subplots()
    plot_ax_data_with_labels(X, y, ax)
    plot_separator(ax, w, b)
    plt.savefig(figname)
    plt.show()
    return fig, ax
