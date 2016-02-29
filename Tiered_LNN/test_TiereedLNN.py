from unittest import TestCase
import numpy as np
from TieredLayeredNN import Layer, NodeClassifier, TieredLayeredNeuralNetwork

# constants
feature_count = 5
functions_in_layer = 8
samples_count = 100
X = np.random.rand(samples_count, feature_count)
y = np.random.randint(0, high=2, size=samples_count)


class TestLayer(TestCase):

    def test_predict_0_returns_same(self):
        layer = Layer(0, NodeClassifier('test layer'))
        X = np.random.randn(3, 3)
        X2 = layer.predict(X)
        self.assertEqual(X.__hash__, X2.__hash__)

    def test_predict_1_returns_correct_shape(self):
        layer = Layer(1)
        for i in range(functions_in_layer):
            classifier = NodeClassifier('extra node' + str(i))
            X = np.random.randn(samples_count, feature_count)
            y = np.random.randint(0, high=2, size=samples_count)
            classifier.fit(X, y)
            layer.nodes.append(classifier)
        x_transformed = layer.predict(X)
        self.assertEqual(x_transformed.shape, (samples_count, functions_in_layer))


class TestNodeClassifier(TestCase):
    def test_init(self):
        node = NodeClassifier('test function')
        self.assertTrue(node.label)

    def test_fit(self):
        node = NodeClassifier('test function')
        node.function.fit(X, y)
        self.assertTrue(True)

    def test_predict(self):
        node = NodeClassifier('test function')
        X = np.random.rand(3, 3)
        y = np.random.randint(0, high=2, size=3)
        node.function.fit(X, y)
        yp = node.function.predict(X)
        self.assertIsNotNone(yp)


class TestMainClassifier(TestCase):
    def setUp(self):
        self.main_classifier = TieredLayeredNeuralNetwork(feature_count)

    def test_activate(self):
        self.fail()

    def test_fit(self):
        self.main_classifier.fit(X, y, 'test main classifier')
        self.assertEqual(len(self.main_classifier.get_labels_list()), 2)

    def test_fit_unequal_width(self):
        self.main_classifier.fit(X, y, 'test main classifier')
        self.assertEqual(len(self.main_classifier.get_labels_list()), 2)
        new_X = np.random.randn(samples_count,42)
        self.main_classifier.fit(X, y, 'unequal width classifier')
        self.assertEqual(len(self.main_classifier.get_labels_list()), 3)




    def test_predict(self):
        self.fail()

    def test_find_label_position(self):
        self.fail()

    def test_score(self):
        self.fail()

    def test_status(self):
        self.fail()
