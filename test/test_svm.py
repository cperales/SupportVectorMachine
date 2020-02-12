from svm import LinearSVM, KernelSVM
from util.generate_data import generate_data
from util.metric import accuracy
import unittest


class TestLinearSVM(unittest.TestCase):

    def test_linear(self):
        clf = LinearSVM()
        X, y = generate_data()
        clf.fit(X=X, y=y, soft=True)
        acc = accuracy(clf=clf, X=X, y=y)
        margin = 0.1
        average = 0.75
        condition = average - margin <= acc <= average + margin
        self.assertEqual(condition, True)

    def test_kernel(self):
        clf = KernelSVM()
        X, y = generate_data()
        clf.fit(X=X, y=y, kernel_type='rbf', k=1.0)
        acc = accuracy(clf=clf, X=X, y=y)
        margin = 0.1
        average = 0.75
        condition = average - margin <= acc <= average + margin
        self.assertEqual(condition, True)

