from unittest import TestCase
import numpy as np
from TieredLayeredNN import Layer, NodeClassifier


class TestLayer(TestCase):
    def setUp(self):
        self.feature_count = 5
        self.functions_in_layer = 8
        self.samples_count = 100

    def test_predict_0_returns_same(self):
        layer = Layer(0, NodeClassifier('test layer'))
        X = np.random.randn(3, 3)
        X2 = layer.predict(X)
        self.assertEqual(X.__hash__, X2.__hash__)

    def test_predict_1_returns_correct_shape(self):
        layer = Layer(1)
        for i in range(self.functions_in_layer):
            classifier = NodeClassifier('extra node' + str(i))
            X = np.random.randn(self.samples_count, self.feature_count)
            y = np.random.randint(0, high=2, size=self.samples_count)
            classifier.fit(X, y)
            layer.nodes.append(classifier)
        x_transformed = layer.predict(X)
        self.assertEqual(x_transformed.shape, (self.samples_count, self.functions_in_layer))


class TestNodeClassifier(TestCase):
    def test_init(self):
        node = NodeClassifier('test function')
        self.assertTrue(node.label)

    def test_fit(self):
        node = NodeClassifier('test function')
        X = np.random.rand(30, 30)
        y = np.random.randint(0, high=2, size=30)
        node.function.fit(X, y)
        self.assertTrue(True)

    def test_predict(self):
        node = NodeClassifier('test function')
        X = np.random.rand(3, 3)
        y = np.random.randint(0, high=2, size=3)
        node.function.fit(X, y)
        yp = node.function.predict(X)
        self.assertIsNotNone(yp)
