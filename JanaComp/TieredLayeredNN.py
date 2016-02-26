'''
Implementing the idea of tiered layered neural network.
'''
import numpy as np
from sklearn.svm import SVC


class NodeClassifier:
    def __init__(self, label):
        """
        :type label: str, a name for the function
        """
        self.function = SVC()
        self.label = label

    def fit(self, X, y):
        """
        :param X: np.array, a matrix of size (samples, features)
        :param y: np.array, a column vector of size (samples, 1)
        """
        self.function.fit(X, y)

    def predict(self, X):
        """
        :param X: np.array, a matrix of size (samples, features)
        """
        return self.function.predict(X)


class Layer:
    def __init__(self, level, function):
        self.nodes = [function]
        self.level = level

    def predict(self, X):
        if self.level == 0:
            return X
        else:
            return np.hstack([node_i.predict(X) for node_i in self.nodes])


class MainClassifier:
    def __init__(self, input_dimension):
        self.layers = [Layer(0, None)]

    def activate(self, X):
        """
        :rtype: np.array, a matrix (layers, samples, features)
        :param X: np.array, a matrix of size (samples, features)
        """
        x_transformed = []
        for (i, layer_i) in enumerate(self.layers):
            if i == 0:
                x_i_transformed = X
            else:
                x_i_transformed = layer_i.predict(x_i_transformed)
            x_transformed.append(x_i_transformed)
        return x_i_transformed

    def fit(self, X, y, label):
        """
        :param label: str, the name of the label
        :param X: np.array, a matrix of size (samples, features)
        :param y: np.array, a column vector of size (samples, 1)
        """
        x_transformed = self.activate(X)
        accuracy_scores = []
        potential_functions = []
        for X_i in x_transformed:
            new_classifier = NodeClassifier(label)
            new_classifier.function.fit(X_i, y)
            accuracy_scores.append(new_classifier.function.score(X_i))
            potential_functions.append(new_classifier)
        best_layer = np.argmax(accuracy_scores)
        layer_to_add_node = best_layer + 1
        chosen_function = potential_functions[best_layer]
        print 'Best fit at level ', best_layer, ' at accuracy score ', accuracy_scores[best_layer]

        if layer_to_add_node > len(self.layers):
            self.layers.append(Layer(level=layer_to_add_node, function=chosen_function))
        else:
            self.layers[best_layer].nodes.append(chosen_function)

    def predict(self, X):
        """
        :param X: np.array, a matrix of size (samples, features)
        """
        x_transformed = self.activate(X)
