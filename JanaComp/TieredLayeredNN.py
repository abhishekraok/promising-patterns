'''
Implementing the idea of tiered layered neural network.
'''
import numpy as np
from sklearn.metrics import f1_score

import School
from PyDataSetSchool import PyDatasetSchool
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
        return self.function.predict(X).reshape(-1,1)


class Layer:
    def __init__(self, level, function):
        self.nodes = [function]
        self.level = level

    def predict(self, X):
        """
        :param X: np.array, size (samples, functions)
        """
        if self.level == 0:
            return X
        else:
            return np.hstack([node_i.predict(X) for node_i in self.nodes])


class MainClassifier:
    def __init__(self, input_dimension):
        self.layers = [Layer(0, NodeClassifier('Raw Input Layer'))]

    def activate(self, X):
        """
        :rtype: list, of length samples, where each element is an np.array of length number_of_functions
        :param X: np.array, a matrix of size (samples, features)
        """
        x_transformed = []
        for (i, layer_i) in enumerate(self.layers):
            if i == 0:
                x_i_transformed = X
            else:
                x_i_transformed = layer_i.predict(x_i_transformed)
            x_transformed.append(x_i_transformed)
        return x_transformed

    def fit(self, X, y, classifier_name):
        """
        :param classifier_name: str, the name of the label
        :param X: np.array, a matrix of size (samples, features)
        :param y: np.array, a column vector of size (samples, 1)
        """
        x_transformed = self.activate(X)
        accuracy_scores = []
        potential_functions = []
        for X_i in x_transformed:
            new_classifier = NodeClassifier(classifier_name)
            new_classifier.function.fit(X_i, y)
            accuracy_scores.append(new_classifier.function.score(X_i, y))
            potential_functions.append(new_classifier)
        best_layer = np.argmax(accuracy_scores)
        layer_to_add_node = best_layer + 1
        chosen_function = potential_functions[best_layer]
        print 'Best fit at level ', best_layer, ' at accuracy score ', accuracy_scores[best_layer]

        if layer_to_add_node >= len(self.layers):
            self.layers.append(Layer(level=layer_to_add_node, function=chosen_function))
        else:
            self.layers[layer_to_add_node].nodes.append(chosen_function)

    def predict(self, X, classifier_name):
        """
        :param X: np.array, a matrix of size (samples, features)
        """
        required_layer_index, required_function_index = self.find_label_position(classifier_name)
        x_transformed = self.activate(X)
        return x_transformed[required_layer_index][:, required_function_index]


    def find_label_position(self, classifier_name):
        for i, layer_i in enumerate(self.layers):
            for j, function_j in enumerate(layer_i.nodes):
                if function_j.label == classifier_name:
                    return i,j
        print 'Function ', classifier_name, ' not found'
        raise RuntimeError

    def score(self, input_x, y, classifier_name):
            yp_score = self.predict(input_x, classifier_name)
            return f1_score(y, y_pred=yp_score)

if __name__ == '__main__':
    main_classifier = MainClassifier(input_dimension=2)
    # School.task_and(main_classifier, labeled_prediction=True)
    PyDatasetSchool.train_iris_setosa(main_classifier, True)

