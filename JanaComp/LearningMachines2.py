""" Version 3 of IRC (Infinite Recursive classifier). Based on the idea that each output is placed in a certain
location.
Let me try to solve a simpler problem first. Let me forget about the gate and do non stop recursive classification
step by step, one bye one.
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
    """ A node that contains classifier, it's input address and output address.
    """
    def __init__(self, end_in_address, out_addres, X=None):
        self.classifier = svm.LinearSVC(dual=False, penalty='l1')
        self.out_address = out_addres
        self.end_in_address = end_in_address # end column

    def fit(self, X, y):
        new_X = X[:,:self.end_in_address]
        self.classifier.fit(new_X,y)
        
    def predict(self, X):
        new_X = X[:,:self.end_in_address]
        return self.classifier.predict(new_X)

class SimpleClassifierBank:
    """ A machine which stores both input X and the current output of bunch of classifiers.
    API should be similar to scikit learn"""

    def __init__(self, max_width, input_width, height=maximum_input_sample_size):
        """
        Initialize this class.

        :rtype : object self
        :param max_width: maximum data dimension in current working memory, should be greater than 
            input_width.
        :param input_width: maximum input dimension. 
        :param height: maximum number of input samples
        :return: None
        """
        self.current_working_memory = np.zeros([height,max_width])
        self.classifiers_out_address_start = input_width
        self.classifiers_current_count = 0  # starting address for ouput for new classifier
        self.classifiers_list = []

    def predict(self, X):
        """Give out what it thinks from the input. Input X should be 2 dimensional.

        :param: X: input, dimensionn 2, (samples x dimension)"""
        X = np.array(X)
        if len(X.shape) is not 2:
            print "Error in predict. Input dimension should be 2"
            raise ValueError
        self.current_working_memory[:X.shape[0], :X.shape[1]] = X
        for classifier_i in self.classifiers_list:
            self.current_working_memory[:,classifier_i.out_address] = classifier_i.predict(self.current_working_memory)
        return self.current_working_memory

    def fit(self, X, y):
        """
        Adds a new classifier and trains it, similar to Scikit API

        :param X: 2d Input data
        :param y:  labels
        :return: None
        """
        X = np.array(X)
        if len(X.shape) is not 2:
            print "Error in predict. Input dimension should be 2"
            raise ValueError
        self.current_working_memory[:X.shape[0], :X.shape[1]] = X

        new_classifier = ClassifierNode(end_in_address=self.classifiers_out_address_start,
                                        out_addres=self.classifiers_out_address_start + 1)
        self.classifiers_out_address_start += 1
        new_classifier.fit(self.current_working_memory, y)
        self.classifiers_list.append(new_classifier)

    def status(self):
        """Gives out the current status, like number of classifier and prints their values"""
        print 'Currently there are ', len(self.classifiers_list), ' classifiers. They are'
        for classifier_i in self.classifiers_list:
            print 'Classifier: ', classifier_i
            print 'Out address', classifier_i.out_address
            print 'In address', classifier_i.end_in_address

Main_C1 = SimpleClassifierBank(max_width=8,input_width=4, height=8)
X = np.array([[0,0], [0,1], [1,0], [0,1]]*2)
y = [0,0,1,1,0,0,1,1]
print X.shape
yp = Main_C1.predict(X)
print yp
print Main_C1.status()
Main_C1.fit(X,y)
print Main_C1.status()
yp = Main_C1.predict(X)
print 'Prediction is '
print yp

Main_C1.fit(X,y)
print Main_C1.status()
yp = Main_C1.predict(X)
print 'Prediction is '
print yp
