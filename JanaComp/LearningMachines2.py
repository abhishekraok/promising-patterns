""" Version 3 of IRC (Infinite Recursive classifier). Based on the idea that each output is placed in a certain
location.
Let me try to solve a simpler problem first. Let me forget about the gate and do non stop recursive classification
step by step, one bye one.

Update. 19 May 2015. Let me stept this up. Instead of having a fixed width,
TODO: let me keep expanding the width. Only the
input width is fixed.
"""
from docutils.nodes import classifier

__author__ = 'Abhishek Rao'

# Headers
import numpy as np
from sklearn import svm
import math
import matplotlib.pyplot as plt
import pickle
import os.path

# Constants

# Global functions
def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def classifier_load(filename='DefaultClassifier.pkl'):
    pickle.load(filename)


# Classes
class ClassifierNode:
    """ A node that contains classifier, it's input address and output address.
    """
    def __init__(self, end_in_address, out_addres, X=None, classifier_name='Default'):
        self.classifier = svm.LinearSVC(dual=False, penalty='l1')
        self.out_address = out_addres
        self.end_in_address = end_in_address # end column
        self.label = classifier_name  # The name of this concept. e.g. like apple etc.

    def fit(self, X, y):
        new_X = X[:,:self.end_in_address]
        self.classifier.fit(new_X,y)
        
    def predict(self, X):
        """
        Give output for the current classifier. Note instead of predict 1,0, better to use probability, soft prediction.
        :param X: The Classifier banks working memory, full matrix.
        :return: A column of predicted values.
        """
        new_X = X[:,:self.end_in_address]
        dec_fx = self.classifier.decision_function(new_X)
        # Convert it into mapping between 0 to 1 instead of -1 to 1
        return np.array([sigmoid(i) for i in dec_fx])


class SimpleClassifierBank:
    """ A machine which stores both input X and the current output of bunch of classifiers.
    API should be similar to scikit learn"""

    def __init__(self, max_width, input_width, heigh):
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
        input_number_samples, input_feature_dimension = X.shape
        if len(X.shape) is not 2:
            print "Error in predict. Input dimension should be 2"
            raise ValueError
        self.current_working_memory[:input_number_samples, :input_feature_dimension] = X
        for classifier_i in self.classifiers_list:
            self.current_working_memory[:,classifier_i.out_address] = classifier_i.predict(self.current_working_memory)
        # need to return the rightmost nonzero column.
        for column_j in range(self.current_working_memory.shape[1]): # traverse through columns
            if np.any(self.current_working_memory[:input_number_samples,column_j]):
                return self.current_working_memory[:input_number_samples,column_j]
        print 'Cant find any nonzero column'
        return self.current_working_memory[:,0]

    def fit(self, X, y, task_name='Default'):
        """
        Adds a new classifier and trains it, similar to Scikit API

        :param X: 2d Input data
        :param y:  labels
        :return: None
        """
        # check for limit reach for number of classifiers.
        if self.classifiers_current_count + self.classifiers_out_address_start \
                > self.current_working_memory.shape[1]:
            print 'No more space for classifier. ERROR'
            raise MemoryError
        X = np.array(X)
        input_number_samples, input_feature_dimension = X.shape
        if len(X.shape) is not 2:
            print "Error in predict. Input dimension should be 2"
            raise ValueError
        self.current_working_memory[:X.shape[0], :X.shape[1]] = X
        # Procure a new classifier, this might be wasteful, later perhaps reuse classifier
        # instead of lavishly getting new ones, chinese restaurant?
        new_classifier = ClassifierNode(end_in_address=self.classifiers_out_address_start,
                                        out_addres=self.classifiers_out_address_start + 1, classifier_name=task_name)
        self.classifiers_out_address_start += 1
        # Need to take care of mismatch in length of working memory and input samples.
        new_classifier.fit(self.current_working_memory[:input_number_samples], y)
        self.classifiers_list.append(new_classifier)

    def status(self):
        """Gives out the current status, like number of classifier and prints their values"""
        print 'Currently there are ', len(self.classifiers_list), ' classifiers. They are'
        for classifier_i in self.classifiers_list:
            print 'Classifier: ', classifier_i
            print 'Classifier name: ', classifier_i.label
            print 'Out address', classifier_i.out_address
            print 'In address', classifier_i.end_in_address
            print 'Coefficients: ', classifier_i.classifier.coef_, classifier_i.classifier.intercept_
        plt.imshow(self.current_working_memory,interpolation='none',cmap='gray')
        plt.title('Current working memory')
        plt.show()

if __name__ == '__main__':
    classifier_file_name = 'ClassifierFile.pkl'
    if os.path.isfile(classifier_file_name):
        Main_C1 = pickle.load(open(classifier_file_name,'r'))
    else:
        Main_C1 = SimpleClassifierBank(max_width=200,input_width=150,height=200)

    # Task 1 noisy and
    noise_columns = np.random.randn(90, 3)
    data_columns = np.array([[1,0], [0,1], [1,1], [0,0]]*2 + [[0,1]])
    data_columns_big = np.vstack([data_columns]*10)
    X = np.hstack([noise_columns, data_columns_big]) # total data
    print X
    y = np.array([[0, 0,1,0]*2 + [0]])
    y_big = np.hstack([y]*10).flatten()
    Main_C1.fit(X,y_big,task_name='Noisy and long')
    Main_C1.status()
    yp = Main_C1.predict(X)
    print 'Predicted value is '
    print yp

    # save the classifier
    pickle.dump(Main_C1,open(classifier_file_name,'w'))