'''
Implementing the idea of tiered layered neural network.

Todo 27 Feb 2016:
1. Unit tests, 2
2. Identify
3. Save Load
4. Caffe front end
'''
import numpy as np
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from PyDataSetSchool import PyDatasetSchool
from sklearn import svm

# constants
svm_c = 1


class NodeClassifier:
    def __init__(self, label):
        """
        :type label: str, a name for the function
        """
        self.function = svm.LinearSVC(C=svm_c, dual=False, penalty='l1')
        self.label = label
        self.width = 0

    def __repr__(self):
        return self.label + ' NodeClassifier'

    def fit(self, X, y):
        """
        :param X: np.array, a matrix of size (samples, features)
        :param y: np.array, a column vector of size (samples, 1)
        """
        self.width = X.shape[1]
        self.function.fit(X, y)

    def predict(self, X):
        """
        :param X: np.array, a matrix of size (samples, features)
        """
        sample_count = X.shape[0]
        input_width = X.shape[1]
        x_correct_shape = np.zeros([sample_count, self.width])
        if input_width > self.width:
            x_correct_shape = X[:,:self.width]
        else:
            x_correct_shape[:,:input_width] = X
        return self.function.predict(x_correct_shape).reshape(-1, 1)


class Layer:
    def __init__(self, level, function=None):
        self.nodes = [function] if function else []
        self.level = level

    def __repr__(self):
        return 'Layer ' + str(self.level) + ' with ' + str(len(self.nodes)) + ' nodes.'

    def predict(self, X):
        """
        :param X: np.array, size (samples, functions)
        """
        if self.level == 0:
            return X
        else:
            return np.hstack([node_i.predict(X) for node_i in self.nodes])


class TieredLayeredNeuralNetwork:
    def __init__(self, input_dimension):
        self.input_dimension = input_dimension
        self.layers = [Layer(0, NodeClassifier('Raw Input Layer'))]
        self.labels_list = ['Raw Input Layer']

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
        if X.shape[1] > self.input_dimension:
            print 'Oye, input dimension is ', X.shape[1], ' is greater than input dimension ', self.input_dimension
            raise BaseException
        x_transformed = self.activate(X)
        accuracy_scores = []
        potential_functions = []
        for X_i in x_transformed:
            new_classifier = NodeClassifier(classifier_name)
            new_classifier.fit(X_i, y)
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
                    return i, j
        print 'Function ', classifier_name, ' not found'
        raise RuntimeError

    def score(self, input_x, y, classifier_name):
        yp_score = self.predict(input_x, classifier_name)
        return f1_score(y, y_pred=yp_score)

    def get_labels_list(self):
        self.labels_list = [function.label for layer in self.layers for function in layer.nodes]
        return self.labels_list

    def status(self, show_graph, show_list):
        """Gives out the current status, like number of functions and prints their values
        :param show_list: Whether to list all the functions
        :param show_graph: Boolean, Whether to show the weights matrix.
        """
        total_number_functions = sum([len(layer_i.nodes) for layer_i in self.layers])
        print 'Currently there are ', len(self.layers), ' layers'
        classifiers_coefficients = np.zeros([total_number_functions, self.input_dimension])
        self.labels_list = self.get_labels_list()
        classifier_count = 0
        if show_list:
            print self.labels_list
        for i, layer_i in enumerate(self.layers):
            if i == 0:
                continue
            for node_i in layer_i.nodes:
                coeffs_i = node_i.function.coef_
                classifiers_coefficients[classifier_count, :coeffs_i.shape[1]] = coeffs_i
                classifier_count += 1

        if show_graph:
            fig, ax = plt.subplots()
            fig.canvas.draw()
            labels = [item.get_text() for item in ax.get_yticklabels()]
            for i, label_i in enumerate(self.labels_list):
                labels[i] = label_i
            plt.imshow(classifiers_coefficients, interpolation='none', cmap='gray')
            ax.set_yticklabels(labels)
            plt.show()


if __name__ == '__main__':
    main_classifier = TieredLayeredNeuralNetwork(input_dimension=10)
    PyDatasetSchool.train_iris(main_classifier, True)
    main_classifier.status(show_graph=True, show_list=True)
    PyDatasetSchool.show_types(100)
