from unittest import TestCase
import sys
import numpy as np

sys.path.append('..')
from Tiered_LNN.TieredLayeredNN import NodeClassifier


class TestNodeClassifier(TestCase):
    def test_init(self):
        node = NodeClassifier('test function')
        self.assertTrue(node.label)

    def test_fit(self):
        node = NodeClassifier('test function')
        X = np.random.rand(3, 3)
        y = np.random.randint(0, high=2, size=3)
        node.function.fit(X, y)
        self.assertTrue(True)

    def test_predict(self):
        node = NodeClassifier('test function')
        X = np.random.rand(3, 3)
        y = np.random.randint(0, high=2, size=3)
        node.function.fit(X, y)
        yp = node.function.predict(X)
        self.assertIsNotNone(yp)
