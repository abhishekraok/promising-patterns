""" Contains class that maintains set of classifiers.
"""
from sklearn import svm
import numpy as np

raw_input_ = 'raw_input_'


class ClassifierFunction:
    def __init__(self, function=None):
        self.main_function = function
        self.inputs = []
        self.collected_input = np.array([])
        self.input_dimension = 0
        self.name = 'DefaultFunction'
        self.out = 0
        self.raw_input_index = 0

    def fit(self, X, y):
        if X.shape[1] != self.input_dimension:
            print 'Invalid input dimension'
            raise
        self.collected_input = self.collect(X)
        self.main_function = svm.SVC()
        self.main_function.fit(self.collected_input, y)

    def predict(self, X):
        if self.raw_input_index >= 0:
            return X[self.raw_input_index]
        else:
            self.collected_input = self.collect(X)
            return self.main_function.predict(self.collected_input)

    def collect(self, X):
        x_transformed = []
        for row in X:
            current_row = []
            for column in self.inputs:
                current_row.append(column.predict(row))
            x_transformed.append(np.hstack(current_row))
        return np.vstack(x_transformed)


class ClassifierManager:
    def __init__(self, input_dimension):
        self.functions_list = [ClassifierFunction(lambda x: x) for i in range(input_dimension)]

    def fit(self, X, y, name=''):
        pass

    def predict(self, X):

    def get_predictor(self):


if __name__ == '__main__':
    X = np.array([[0, 1], [1, 0], [0, 1]])
    y = np.array([0, 1, 0])
    cm = ClassifierManager([X, y])
    cm.fit(X, y)
