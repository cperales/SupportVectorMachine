from svm import LinearSVM, KernelSVM
from util.generate_data import generate_data, plot_data_separator
from util.metric import accuracy
import logging
import numpy as np

try:
    import os

    os.remove('images/svm.png')
    logging.debug('Previous image was removed')
except:
    logging.debug('Previous image did not exit')

X, y = generate_data(dataname='../gaussiandata.pickle')
clf = LinearSVM()
clf.fit(X=X, y=y, soft=True)
y_pred = clf.predict(X=X)
acc = accuracy(clf, X=X, y=y, y_pred=y_pred)
logging.info('Accuracy (for training) = {}'.format(acc))
plot_data_separator(X, y, clf.w, clf.b, '../svm.png')


clf = KernelSVM()
clf.fit(X=X, y=y, kernel_type='rbf', k=1.0)
acc = accuracy(clf, X=X, y=y)
logging.info('Accuracy (for training) kernel rbf = {}'.format(acc))

