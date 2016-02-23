""" Contains class that maintains set of classifiers.
"""
from sklearn import svm
import numpy as np


class SpecialClassifier:
    def __init__(self, main_classifier=None):
        self.main_classifier = main_classifier
        self.input_transformer = lambda x: x

    def fit(self, X, y):
        x_transformed = self.input_transformer(X)
        self.main_classifier.fit(x_transformed, y)

    def predict(self, X):
        x_transformed = self.input_transformer(X)
        self.main_classifier.predict(x_transformed)


class ClassifierManager:
    def __init__(self, input_xy=None):
        self.classifier_list = []
        if input_xy is not None:
            clf = SpecialClassifier(svm.SVC())
            clf.fit(input_xy[0], input_xy[1])
            self.classifier_list.append(clf)

    def fit(self, X, y):
        x_transformed = self.predict(X)
        clf = SpecialClassifier(svm.SVC())
        clf.input_transformer = self.predict
        clf.fit(x_transformed, y)
        clf.input_transformer = self.get_predictor()
        self.classifier_list.append(clf)

    def predict(self, X):
        fn = self.get_predictor()
        b = fn(X)
        return b

    def get_predictor(self):
        def predictor(X):
            return np.hstack([y.predict(X) for y in self.classifier_list])

        return predictor


if __name__ == '__main__':
    X = np.array([[0, 1], [1, 0], [0, 1]])
    y = np.array([0, 1, 0])
    cm = ClassifierManager([X, y])
    cm.fit(X, y)
