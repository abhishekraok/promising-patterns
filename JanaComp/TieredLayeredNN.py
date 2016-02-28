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
        self.input_dimension = input_dimension
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
        if X.shape[1] != self.input_dimension:
            print 'Oye, input dimension is ', X.shape[1], ' and I am made for ', self.input_dimension
            raise BaseException
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


    def status(self, show_graph, show_list):
        """Gives out the current status, like number of functions and prints their values
        :param show_list: Whether to list all the functions
        :param show_graph: Boolean, Whether to show the weights matrix.
        """
        total_number_functions = sum([len(layer_i.nodes) for layer_i in self.layers])

        print 'Currently there are ', len(self.layers), ' layers'
        classifiers_coefficients = np.zeros([total_number_functions, self.input_dimension + 1])
        self.labels_list = [function.label for layer in self.layers for function in layer.nodes]
        if show_list:
            print self.labels_list
        # for count, classifier_i in enumerate(self.classifiers_list):
        #     coeffs_i = classifier_i.classifier.coef_ \
        #         if classifier_i.classifier_type == 'standard' else np.zeros([1, 1])
        #     classifiers_coefficients[count, :coeffs_i.shape[1]] = coeffs_i
        #     #    print 'Classifier: ', classifier_i
        #     #    print 'Classifier name: ', classifier_i.label
        #     #    print 'Out address', classifier_i.out_address
        #     #    print 'In address', classifier_i.end_in_address
        #     # print 'Coefficients: ', classifier_i.classifier.coef_, classifier_i.classifier.intercept_
        # if show_graph:
        #     decimation_factor = int(self.current_working_memory.shape[0] / 40) + 1
        #     plt.figure()
        #     plt.imshow(self.current_working_memory[::decimation_factor,
        #                self.prediction_column_start:
        #                self.memory_width], interpolation='none', cmap='gray')
        #     plt.title('Current working memory')
        #     # Coefficients matrix plot
        #     plt.figure()
        #     plt.imshow(classifiers_coefficients, interpolation='none', cmap='gray')
        #     plt.title('Classifier coefficients')
        #     plt.figure()
        #     plt.imshow(classifiers_coefficients[:, self.prediction_column_start:], interpolation='none', cmap='gray')
        #     plt.title('Classifier interdependency')
        #     plt.show()
        #
        #     # Coefficients matrix plot, sparsity, thresholded.
        #     plt.figure()
        #     classifiers_coefficients[classifiers_coefficients != 0] = 1
        #     # The mean number of non zero coefficients, 2 because its triangular.
        #     print 'Sparsity ratio is ', 2 * classifiers_coefficients.mean()
        #     plt.imshow(classifiers_coefficients, interpolation='none', cmap='gray')
        #     plt.title('Sparsity of coefficients')
        #     # classifier interdependency.
        #     plt.show()
        #     return 2 * classifiers_coefficients.mean()
        # return 0

if __name__ == '__main__':
    main_classifier = MainClassifier(input_dimension=4)
    # School.task_and(main_classifier, labeled_prediction=True)
    PyDatasetSchool.train_iris_setosa(main_classifier, True)
    main_classifier.status(show_graph=False, show_list=True)

