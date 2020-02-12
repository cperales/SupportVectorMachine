from svm import LinearSVM, KernelSVM
from util.generate_data import generate_data, plot_data_separator, \
    train_test_split
from util.metric import accuracy
import logging
import numpy as np

X, y = generate_data(dataname='../gaussiandata.pickle')
X_train, y_train, X_test, y_test = train_test_split(X, y, prop=0.25)
# try:
#     import os
#
#     os.remove('images/svm.png')
#     logging.debug('Previous image was removed')
# except:
#     logging.debug('Previous image did not exit')
#
clf = LinearSVM()
clf.fit(X=X_train, y=y_train, soft=True)
acc_train = accuracy(clf, X=X_train, y=y_train)
acc_test = accuracy(clf, X=X_test, y=y_test)
logging.info('Accuracy (on training) = {}'.format(acc_train))
logging.info('Accuracy (on test) = {}'.format(acc_test))
# plot_data_separator(X, y, clf.w, clf.b, '../svm.png')


clf = KernelSVM()
clf.fit(X=X_train, y=y_train, kernel_type='rbf', k=1.0)
acc_train = accuracy(clf, X=X_train, y=y_train)
logging.info('Accuracy (on training) kernel rbf = {}'.format(acc_train))
acc_test = accuracy(clf, X=X_test, y=y_test)
logging.info('Accuracy (on testing) kernel rbf = {}'.format(acc_test))

