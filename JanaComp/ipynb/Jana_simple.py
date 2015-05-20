__author__ = 'Abhishek Rao'

import math
import numpy as np
import random


class Jana():
    def __init__(self, start_cupee=10, name="Default"):
        """Initialize a Jana, with some Cuppe and no weights(connections)."""
        self.Cupee = start_cupee
        self.connections = []  # Integer index in jana_list
        self.weights = np.array([])
        self.name = name
        self.output = 0

    def activate(self, calling_loka):
        """Based on the input, set the output"""
        input_jana = np.array([calling_loka.jana_list[k].output for k in self.connections])
        self.output = np.dot(input_jana, self.weights)


class Loka():
    """A world of janas."""

    def __init__(self, names_list=None):
        if not names_list:
            names_list = ['Gamma', 'Soma', 'Bhima', 'Rama']
        self.jana_list = [Jana(name=i) for i in names_list]
        self.number_of_janas = len(self.jana_list)
        self.learning_rate = 0.01

    def get_names(self):
        """Lists all the names of each jana"""
        for i in self.jana_list:
            print i.name

    def get_status_all(self):
        """Lists all the names, cupees and connections of each jana"""
        for i in self.jana_list:
            print 'Name =', i.name, ' Cupee = ', i.Cupee, ' Connections = ', i.connections, \
                ' Weights = ', i.weights, 'Ouput = ', i.output

    def initialize_connections(self):
        """Randomly Set the connections if their connection is []"""
        for i in self.jana_list:
            if not i.connections:
                # Choose the number of connections as Poisson ~ (sqrt(total janas))
                num_connections = np.random.poisson(int(math.sqrt(self.number_of_janas)), 1)[0]
                # Prevent 0s
                num_connections = min(max(num_connections, 1), self.number_of_janas)
                # Randomly choose these number of connections
                i.connections = random.sample(range(self.number_of_janas), num_connections)
                # initialize them with weight N(0,1)
                i.weights = np.random.normal(0, 1, num_connections)

    def train(self, X, y):
        """Given an input X, and output y, we use a Hebbian rule update to
        move the weight more towards predicting """
        sample_count = 0
        for ith_sample in X:  # For every sample (row of X)
            jana_count = 0  # counter to keep track of how many jana used
            # first fix X, then y
            for xij in ith_sample:  # for each dimension (column of X)
                self.jana_list[jana_count].output = xij
                jana_count += 1
            self.jana_list[-1].output = y[sample_count]  # set the last one to y.
            # self.get_status_all()
            # hebbian weight update
            for jana_i in self.jana_list:
                input_jana = np.array([self.jana_list[k].output for k in jana_i.connections])
                # print 'current Jana = ',jana_i.name, 'Input jana = ', input_jana
                jana_i.weights += self.learning_rate * (input_jana * jana_i.output)
        sample_count += 1

    def predict(self, X):
        """Given an input X, guess the output y
        """
        sample_count = 0
        y = []
        for ith_sample in X:  # For every sample (row of X)
            jana_count = 0  # counter to keep track of how many jana used
            # first fix X, then y
            for xij in ith_sample:  # for each dimension (column of X)
                self.jana_list[jana_count].output = xij
                jana_count += 1
            print 'At predict'
            self.get_status_all()
            for n in range(10):
                for jana_i in self.jana_list:
                    jana_i.activate(self)
            y.append(self.jana_list[-1].output)
        return y


Loka1 = Loka()
Loka1.initialize_connections()
Loka1.get_status_all()
X = [[1, 1], [1, 0], [0, 1], [0, 0]] * 100
y = [1, 0, 0, 0] * 100

Loka1.train(X, y)
Loka1.predict(X[:4])