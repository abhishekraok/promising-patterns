""" This is the place where young classifiers are trained on Pydataset
"""
__author__ = 'Abhishek Rao'

# Headers
import numpy as np
from sklearn.cross_validation import train_test_split
from pydataset import data


class PyDatasetSchool:
    def __init__(self):
        pass

    @staticmethod
    def basic_train(dataset, training_name, classifier, prediction_label=False):
        """
        A skeleton function that does basic training and testing.

        :param dataset: tuple, The X, y pair, X is a numpy matrix (samples, features), y is a np.array
        :param training_name: str, the name of this training.
        :param classifier: should have fit, predict and score functions.
        :param prediction_label: Boolean, whether the classifier accepts a label for train and predict.
        :return:
        """
        X, y = dataset
        x_train, x_test, y_train, y_test = train_test_split(X, y)
        print 'Training for ', training_name, ' with ', x_train.shape, ' training samples and ', x_test.shape, ' test samples'
        if prediction_label:
            classifier.fit(x_train, y_train, training_name)
            score = classifier.score(x_train, y_train, training_name)
        else:
            classifier.fit(x_train, y_train)
            score = classifier.score(x_train, y_train)
        print 'The score for task ', training_name, ' is ', score

    @staticmethod
    def train_iris(classifier, prediction_label):
        classifier_base_name = 'PyDataset_iris_'
        iris = data('iris')
        X = np.array(iris)[:, :4]
        labels = np.array(iris)[:, 4]
        unique_labels = list(set(labels))
        for label_i in unique_labels:
            y = np.array([i == label_i for i in labels])
            PyDatasetSchool.basic_train((X, y),classifier_base_name + label_i , classifier, prediction_label)
