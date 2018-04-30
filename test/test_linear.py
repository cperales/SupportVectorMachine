from svm.linear import LinearSVM
from util.generate_data import generate_data, plot_data_separator
import logging

try:
    import os

    os.remove('images/svm.png')
    logging.debug('Previous image was removed')
except:
    logging.debug('Previous image did not exit')

clf = LinearSVM()
X, y = generate_data(dataname='../data/gaussiandata.pickle')
# X,y = read_data('data/gaussiandata.pickle')
clf.fit(X=X, y=y, soft=True)
plot_data_separator(X, y, clf.w, clf.b, '../images/svm.png')
