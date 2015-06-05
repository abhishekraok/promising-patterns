"""
Title: Remembering Machine Unsupervised

Tries to remember data pattern using cells.
"""

__author__ = 'Abhishek Rao'

# Headers
import numpy as np
# from sklearn import svm
# import math
import matplotlib.pyplot as plt
import cPickle as pickle
# from sklearn.metrics import f1_score
# import sys
import os
# import copy
# import glob
# import time
import gzip
# from random import shuffle
from numpy.random import choice
from numpy import linalg as LA
import sklearn.datasets

# Constants
vector_retention_threshold = 0.1
vector_second_threshold = 0.1
max_input_dimension = 8


# Classes
class LearntVectors:
    """ a vector similar to previously seen input..
    """
    def __init__(self, end_address, out_address):
        self.end_in_address = end_address
        self.activation = np.zeros([1, 1])
        self.weight_vector = np.array([])
        self.out_address = out_address


class UnsupervisedVectorMachines:
    """ contains many learning vectors.

    """

    def __init__(self, input_width):
        self.working_memory = np.zeros([1, input_width])
        self.vectors_list = []
        self.vectors_start_addres = input_width
        self.vectors_end_address = input_width
        new_random_vector = LearntVectors(input_width, input_width)
        new_random_vector.weight_vector = np.random.randn(input_width)/input_width
        self.vectors_list.append(new_random_vector)
        self.vectors_end_address += 1

    def see(self, x_in):
        """
        Take in a new input.
        :param x_in:
        :return:
        """
        x_in = np.array(x_in)
        input_number_samples, input_feature_dimension = x_in.shape
        # Normalize row wise.
        row_wise_energy = LA.norm(x_in, ord=2, axis=1)
        x_in = np.divide(x_in.T, row_wise_energy).T
        x_in -= np.mean(x_in, axis=1)
        # create working memory and load it with input.
        self.working_memory = np.zeros([input_number_samples, self.vectors_end_address])
        self.working_memory[:input_number_samples, :input_feature_dimension] = x_in
        # Start forward pass
        for vector_i in self.vectors_list:
            input_vector = self.working_memory[:, :vector_i.end_in_address]
            output_vector = np.dot(input_vector, vector_i.weight_vector)
            output_vector = np.reshape(output_vector, (-1, 1))
            self.working_memory[:, vector_i.out_address] = output_vector
        # Remove known inputs.
        for vector_i in self.vectors_list:
            activations = self.working_memory[:, vector_i.out_address]
            self.working_memory[:, :vector_i.end_in_address] -= \
                activations * vector_i.weight_vector
        # Now pick one to represent next based on the remaining ones.
        # see which row has highest energy
        rows = self.working_memory.shape[0]
        row_wise_energy = LA.norm(self.working_memory, ord=2, axis=1)
        row_wise_energy /= np.sum(row_wise_energy)
        chosen_row_index = choice(range(rows), p=row_wise_energy)
        chosen_row = self.working_memory[chosen_row_index, :]
        # check if it has more than one element, to see if it's worth learning.
        sorted_values = np.sort(chosen_row)
        if sorted_values[-1] > vector_retention_threshold:
            if sorted_values[-2] > vector_second_threshold:
                self.create_new_vector(self.working_memory[chosen_row_index, :])
                print 'Learnt a new vector.'
            else:
                print 'Nothing new here. This is simlar to one at ', chosen_row.argmax()
        else:
            print 'Nothing is input'

    def create_new_vector(self, weight_vector):
        new_vector = LearntVectors(
            end_address=self.vectors_end_address,
            out_address=self.vectors_end_address )
        new_weight_vector = weight_vector
        # Normalize
        new_weight_vector/= LA.norm(weight_vector, ord=2)
        new_weight_vector -= np.mean(new_weight_vector)
        new_vector.weight_vector = new_weight_vector
        self.vectors_end_address += 1
        self.vectors_list.append(new_vector)

    def save(self, filename):
        pickle.dump(self, gzip.open(filename, 'w'))
        print 'Remembering Machine saved.'

    def status(self, show_graph=False):
        """Gives out the current status, like number of vectors and prints their values"""
        print 'Currently there are ', len(self.vectors_list)
        classifiers_coefficients = np.zeros([len(self.vectors_list), self.vectors_end_address])
        for count, vector_i in enumerate(self.vectors_list):
            coeffs_i = vector_i.weight_vector.reshape(1,-1)
            classifiers_coefficients[count, :coeffs_i.shape[1]] = coeffs_i
        if show_graph:
            plt.imshow(self.working_memory[:200,:], interpolation='none', cmap='gray')
            plt.title('Current working memory')
            plt.figure()
            plt.imshow(classifiers_coefficients, interpolation='none', cmap='gray')
            plt.title('Classifier coefficients')
            plt.show()


if __name__ == '__main__':
    classifier_file_name = 'UnsupervisedVectorMachine.pkl.gz'
    print 'Loading classifier file ...'
    if os.path.isfile(classifier_file_name):
        papu = pickle.load(gzip.open(classifier_file_name, 'r'))
    else:
        papu = UnsupervisedVectorMachines(input_width=max_input_dimension)
    print 'Loading complete.'
    # pattern = np.array([1, 0, 1, 0, 1, 0, 1]).reshape(1, -1)
    # pattern2 = np.array([1, 1, 1, 1, 0, 0, 0]).reshape(1, -1)
    # papu.see(pattern)
    # papu.see(pattern2)
    iris = sklearn.datasets.load_iris()
    x_in = iris.data
    for i in range(1):
        for j in x_in[:3,:]:
            papu.see(j.reshape(1,-1))
    papu.status(show_graph=True)
    papu.save(classifier_file_name)
