""" Version 3 of IRC (Infinite Recursive classifier). Based on the idea that each output is placed in a certain
location.
Let me try to solve a simpler problem first. Let me forget about the gate and do non stop recursive classification
step by step, one bye one.

Update. 19 May 2015. Let me stept this up. Instead of having a fixed width,

Update. 21 May 2015: Split into files, created School.py

#TODO: extending classifier
    let me keep expanding the width. Only the
    Variable width output for classifier.
    Assign any function to a classifier node.
input width is fixed.
"""

__author__ = 'Abhishek Rao'


# Headers
import numpy as np
from sklearn import svm
import math
import matplotlib.pyplot as plt
import pickle
import os.path
from sklearn.metrics import accuracy_score
import School


# Constants


# Classes
class ClassifierNode:
    """ A node that contains classifier, it's input address and output address.
    """

    def __init__(self, end_in_address, out_address, classifier_name='Default',
                 given_predictor=None):
        self.out_address = out_address
        self.end_in_address = end_in_address  # end column
        self.label = classifier_name  # The name of this concept. e.g. like apple etc.
        # Check whether to create a standard classifier or a custom, given one.
        if given_predictor:
            self.given_predictor = given_predictor
            self.classifier_type = 'custom'
        else:
            self.classifier = svm.LinearSVC(dual=False, penalty='l1')
            self.classifier_type = 'standard'

    def fit(self, X, y):
        new_x = X[:, :self.end_in_address]
        self.classifier.fit(new_x, y)

    def predict(self, X):
        """
        Give output for the current classifier. Note instead of predict 1,0, better to use probability, soft prediction.
        :param X: The Classifier banks working memory, full matrix.
        :return: A column of predicted values.
        """
        new_x = X[:, :self.end_in_address]
        if self.classifier_type == 'standard':
            dec_fx = self.classifier.decision_function(new_x)
        else:
            dec_fx = self.given_predictor(new_x)
        # Convert it into mapping between 0 to 1 instead of -1 to 1
        return np.array([sigmoid_10(i) for i in dec_fx])


class SimpleClassifierBank:
    """ A machine which stores both input X and the current output of bunch of classifiers.
    API should be similar to scikit learn"""

    def __init__(self, max_width, input_width, height):
        """
        Initialize this class.

        :rtype : object self
        :param max_width: maximum data dimension in current working memory, should be greater than
            input_width.
        :param input_width: maximum input dimension.
        :param height: maximum number of input samples
        :return: None
        """
        self.current_working_memory = np.zeros([height, max_width])
        self.classifiers_out_address_start = input_width  # the start of classifiers output.
        self.classifiers_current_count = 0  # starting address for output for new classifier
        self.classifiers_list = []

    def predict(self, X):
        """Give out what it thinks from the input. Input X should be 2 dimensional.

        :param: X: input, dimension 2, (samples x dimension)"""
        self.current_working_memory *= 0  # Flush the current input
        X = np.array(X)
        input_number_samples, input_feature_dimension = X.shape
        if len(X.shape) is not 2:
            print "Error in predict. Input dimension should be 2"
            raise ValueError
        self.current_working_memory[:input_number_samples, :input_feature_dimension] = X
        for classifier_i in self.classifiers_list:
            predicted_value = classifier_i.predict(self.current_working_memory)
            predicted_shape = predicted_value.shape
            if len(predicted_shape) < 2:
                predicted_value = predicted_value.reshape(-1, 1)
            predicted_shape = predicted_value.shape
            self.current_working_memory[:predicted_shape[0], classifier_i.out_address] = predicted_value
        # need to return the rightmost nonzero column.
        for column_j in range(self.current_working_memory.shape[1])[::-1]:  # reverse traverse through columns
            if np.any(self.current_working_memory[:input_number_samples, column_j]):
                soft_dec = self.current_working_memory[:input_number_samples, column_j]
                return np.array(soft_dec > 0.5, dtype=np.int16)
        print 'Cant find any nonzero column'
        return self.current_working_memory[:, 0]

    def fit(self, x_in, y, task_name='Default'):
        """
        Adds a new classifier and trains it, similar to Scikit API

        :param x_in: 2d Input data
        :param y:  labels
        :return: None
        """
        # check for limit reach for number of classifiers.
        if self.classifiers_current_count + self.classifiers_out_address_start \
                > self.current_working_memory.shape[1]:
            print 'No more space for classifier. ERROR'
            raise MemoryError
        x_in = np.array(x_in)
        input_number_samples, input_feature_dimension = x_in.shape
        if len(x_in.shape) is not 2:
            print "Error in predict. Input dimension should be 2"
            raise ValueError
        self.current_working_memory[:x_in.shape[0], :x_in.shape[1]] = x_in
        # Procure a new classifier, this might be wasteful, later perhaps reuse classifier
        # instead of lavishly getting new ones, chinese restaurant?
        new_classifier = ClassifierNode(
            end_in_address=self.classifiers_out_address_start + self.classifiers_current_count,
            out_address=[self.classifiers_out_address_start + self.classifiers_current_count + 1],
            classifier_name=task_name)
        self.classifiers_current_count += 1
        # Need to take care of mismatch in length of working memory and input samples.
        new_classifier.fit(self.current_working_memory[:input_number_samples], y)
        self.classifiers_list.append(new_classifier)

    def fit_custom_fx(self, custom_function, input_width, output_width, task_name):
        """
        Push in a new custom function to classifiers list.
        :param custom_function: The function that will be used to predict. Should take in a 2D array input and
            give out a 2d array of same height and variable width.
        :param input_width: The width of input.
        :param output_width: The width of output. If a single neuron this is one.
        :param task_name: name of this function
        :return: None
        """
        new_classifier = ClassifierNode(
            end_in_address=input_width,
            out_address=self.classifiers_out_address_start + self.classifiers_current_count + np.arange(output_width),
            classifier_name=task_name,
            given_predictor=custom_function
        )
        self.classifiers_current_count += output_width
        self.classifiers_list.append(new_classifier)

    def status(self):
        """Gives out the current status, like number of classifier and prints their values"""
        print 'Currently there are ', len(self.classifiers_list), ' classifiers. They are'
        classifiers_coefficients = np.zeros(self.current_working_memory.shape)
        print [classifier_i.label for classifier_i in self.classifiers_list]
        for count, classifier_i in enumerate(self.classifiers_list):
            coeffs_i = classifier_i.classifier.coef_ \
                if classifier_i.classifier_type == 'standard' else np.zeros([1, 1])
            classifiers_coefficients[count, :coeffs_i.shape[1]] = coeffs_i
            #    print 'Classifier: ', classifier_i
            #    print 'Classifier name: ', classifier_i.label
            #    print 'Out address', classifier_i.out_address
            #    print 'In address', classifier_i.end_in_address
            # print 'Coefficients: ', classifier_i.classifier.coef_, classifier_i.classifier.intercept_
        plt.imshow(self.current_working_memory, interpolation='none', cmap='gray')
        plt.title('Current working memory')
        plt.figure()
        plt.imshow(classifiers_coefficients, interpolation='none', cmap='gray')
        plt.title('Classifier coefficients')
        plt.show()

    def remove_classifier(self, classifier_name):
        """
        Removes the classifier whose name is same as classifier_name
        :param classifier_name: the label of the classifier to be removed.
        :return: the index of removed classifier. -1 if not found.
        """
        try:
            labels_list = [classifier_i.label for classifier_i in self.classifiers_list]
        except ValueError:
            print 'The specified label does not exist.'
            return -1
        removing_index = labels_list.index(classifier_name)
        self.classifiers_list.pop(removing_index)
        print 'Classifier was removed. Its nae was', classifier_name
        return removing_index

    def score(self, x_in, y):
        """
        Gives the accuracy between predicted( x_in) and y
        :param x_in: 2d matrix, samples x_in dimension
        :param y: actual label
        :return: float, between 0 to 1
        """
        yp = self.predict(x_in)
        return accuracy_score(y, y_pred=yp)

    def generic_task(self, x_in, y, task_name):
        """
        A generic framework to train on different tasks.
        """
        self.fit(x_in, y, task_name=task_name)
        print 'The score for task ', task_name, ' is ', self.score(x_in, y)


# Global functions
# Reason for having 10 sigmoid is to get sharper distinction.
def sigmoid_10(x):
    return 1 / (1 + math.exp(-10*x))

# Following are required for custom functions Task 1,2
def meanie(x):
    return np.mean(x, axis=1)

def dot_with_11(x):
    return np.dot(x, np.array([0.5, 0.5]))


if __name__ == '__main__':
    classifier_file_name = 'ClassifierFile.pkl'
    if os.path.isfile(classifier_file_name):
        Main_C1 = pickle.load(open(classifier_file_name, 'r'))
    else:
        Main_C1 = SimpleClassifierBank(max_width=2000, input_width=1500, height=500)
    # School.class_digital_logic(Main_C1)
    # Main_C1.fit_custom_fx(np.mean,input_width=1500, output_width=1, task_name='np.mean')
    yp = Main_C1.predict(np.random.randn(8, 22))
    print 'Predicted value is ', yp
    # Main_C1.remove_classifier('np.mean')
    # School.simple_custom_fitting_class(Main_C1)
    Main_C1.status()
    pickle.dump(Main_C1, open(classifier_file_name, 'w'))
