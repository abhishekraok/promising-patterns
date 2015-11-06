"""
Title: Collector Network

Implementing the Collector network classifier. The main idea is to erase the inputs.
Date: 2 Nov 2015

Date: 4 Nov 2015: Let me try linking these collector nodes by reference.
"""
import numpy as np

__author__ = 'Abhishek Rao'


class CollectorNode:
    """
    A collector node. Expects normalized input. Output is one if weighted sum
    above threshold. Similar to neuron in ANN.
    Derived from https://github.com/abhishekraok/promising-patterns/blob/master/JanaComp/RememberingMachine.py
    ClassifierNode

    Dimension of weight vector should be same as number of collectors.
    """

    def __init__(self, output_value=0, input_collectors=[], weight_vector=[]):
        self.output = output_value
        self.confidence = 0
        self.threshold = 0
        self.input_Collectors = input_collectors
        self.weight_vector = np.array(weight_vector)

    def dot(self, another_vector):
        return np.dot(self.weight_vector, another_vector)

    def similar(self, another_vector):
        return self.dot(another_vector) > self.threshold

    def status(self):
        print 'Threshold is ' + str(self.threshold)
        print 'Output value is ' + str(self.output)
        print 'Self confidence is ' + str(self.confidence)
        print 'Weights vector is ' + str(self.weight_vector)

    def activate_input(self):
        """
        Reverse activate, sets all input to 1, sets it's own output to 0.

        Should set input to it's weight?
        """
        self.output = 0
        for input_vector in self.input_Collectors:
            input_vector.output = 1
            input_vector.activate_input()

    def activate_output(self):
        """
        Collects the input and creates an input vector, dots it with own
        weight vector. Calculates both confidence and output.
        """
        input_vector = np.array([i.output for i in self.input_Collectors])
        self.confidence = np.dot(input_vector, self.weight_vector)
        self.output = 1 * (self.confidence > self.threshold)


class CollectorNetwork:
    """
    Collection of collection network.
    """

    def __init__(self, input_dimension=10):
        self.input_collection = [CollectorNode(0)] * 10
        self.collection = []

    # def remember(self, X):
    #     self.activate(X)
    #     new_cn = CollectorNode(weight_vector=X)
    #     self.collection.append(new_cn)

    def activate(self, X):
        """
        Activates all the input CN and then the collection nodes.

        :param X: (1 row, n cols) np array
        """
        input_dimension = X.shape[1]
        for i in range(input_dimension):
            self.input_collection[i].output = X[i]
        for collector_node_i in self.collection:
            collector_node_i.activate_output()




if __name__ == '__main__':
    cn1 = CollectorNode(0)
    cn2 = CollectorNode(3)
    cn3 = CollectorNode(2)
    cn4 = CollectorNode(input_collectors=[cn1, cn2, cn3], weight_vector=[1, 2, 3])
    cn4.status()
    cn4.activate_output()
    cn4.status()
