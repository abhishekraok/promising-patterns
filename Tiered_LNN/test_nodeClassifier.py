from unittest import TestCase
import numpy as np
from TieredLayeredNN import Layer, NodeClassifier


class TestLayer(TestCase):
    def test_predict_0_returns_same(self):
        layer = Layer(0, NodeClassifier('test layer'))
        X = np.random.randn(3, 3)
        X2 = layer.predict(X)
        self.assertEqual(X.__hash__, X2.__hash__)


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
