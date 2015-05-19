""" Version 2 of IRC (Infinite Recursive classifier). Based on the idea that each output is placed in a certain
location.
"""
__author__ = 'Abhishek Rao'

# Headers
import numpy as np
from sklearn import svm

# Constants
maximum_input_dimension = 5
maximum_input_sample_size = 3

# Classes
class ClassifierNode:
    """ A noce that contains a classifier, a gate and address where the
    output is written.
    """
    def __init__(self, in_address, out_addres, X=None):
        self.classifier = svm.SVC()
        self.gate = svm.LinearSVC(dual=False,penalty='l1')
        self.out_address = out_addres
        if X is not None:
            self.data_shape = X.shape
        self.in_address = in_address

    def fit(self, X, y):
        self.classifier.fit(X,y)
        self.data_shape = X.shape
        self.gate.fit()

    def get_activation(self, current_working_memory):
        """
        Says how much of the current input is recognized.
        :param 2d Matrixcurrent_working_memory: the current working
            memory of ThoughtfulMachine class.
        :return: float, score of gate function.
        """
        start_X_height, end_X_height, start_X_width, end_X_width = self.in_address
        new_X = current_working_memory[start_X_height:end_X_height, start_X_width:end_X_width]
        return self.gate.score(new_X)


class ThoughtfulMachine:
    """ A machine which stores both input X and the current output of bunch of classifiers.
    API should be similar to scikit learn"""

    def __init__(self, width=maximum_input_dimension, height=maximum_input_sample_size):
        """
        Initialize this class.

        :rtype : object self
        :param width: maximum data dimension in current working memory
        :param height: maximum number of input samples
        :return: None
        """
        self.current_working_memory = np.zeros([height,width])

    def predict(self, X):
        """Give out what it thinks from the input. Input X should be 2 dimensional.

        :param: X: input, dimensionn 2, (samples x dimension)"""
        X = np.array(X)
        if len(X.shape) is not 2:
            print "Error in predict. Input dimension should be 2"
            raise ValueError
        self.current_working_memory[:X.shape[0], :X.shape[1]] = X
        print self.current_working_memory
        return self.current_working_memory

    def fit_new(self, X, y):
        """
        Adds a new classifier and trains it, similar to Scikit API

        :param X: 2d Input data
        :param y:  labels
        :return: None
        """


    def fit(self, X,y):
        # let's see what the existing classifier have to say about the incoming data.
        X = np.array(X)
        print X.shape
        in_sample_size = X.shape[0]
        if len(X.shape) > 1:
            in_dim = X.shape[1]
        # Assumption, single output, each prediction is a column in the matrix, stack them horizontally
        # to build a matrix
        if  self.classifiers_list:
            self.side_info = self.predict(X)
        # side info should be a tensorr matrix with dimension (number of classifiers, X rows, variable width)
        print 'Side info'
        print self.side_info
        if len(self.side_info) == 0:
            new_X = X
        else:
            new_X = np.vstack([np.hstack([X[j,:], self.side_info[j,:]]) for j in range(in_sample_size)])
        print 'The new X is '
        print new_X
        clf = svm.SVC()
        clf.fit(new_X,y)
        self.classifiers_list.append(clf)
        self.number_of_classifier = len(self.classifiers_list)


    def status(self):
        """Gives out the current status, like number of classifier and prints their values"""
        print 'Currently there are ', len(self.classifiers_list), ' classifiers. They are'
        for classifier_i in self.classifiers_list:
            print classifier_i

Main_C1 = ThoughtfulMachine(height=8, width=5)
X = np.array([[0,0], [0,1], [1,0], [0,1]]*2)
print X.shape
yp = Main_C1.predict(X)
print yp
