""" This is the place where young remembering machine learning classifiers are trained in various arts.
"""
__author__ = 'Abhishek Rao'

# Headers
import numpy as np
from LearningMachines2 import meanie, dot_with_11
from sklearn.cross_validation import train_test_split
import os
import glob

def task_and(classifier):
    # Task 1 noisy and
    noise_columns = np.random.randn(90, 3)
    data_columns = np.array([[1, 0], [0, 1], [1, 1], [0, 0]] * 2 + [[0, 1]])
    data_columns_big = np.vstack([data_columns] * 10)
    X = np.hstack([data_columns_big, noise_columns])  # total data
    print X
    y = np.array([[0, 0, 1, 0] * 2 + [0]])
    y_big = np.hstack([y] * 10).flatten()
    classifier.fit(X, y_big, object_label='Noisy and long')
    yp = classifier.predict(X)
    print 'Predicted value is '
    print yp
    print 'Score for task Noisy and long is ', classifier.score(X, y_big)

def task_XOR_problem(classifier):
    """
    Trains the classifier in the art of XOR problem
    :param classifier: any general classifier.
    :return: None
    """
    X = np.array([[1, 0],
                  [0, 1],
                  [1, 1],
                  [0, 0]] * 50)
    y = np.array([1, 1, 0, 0] * 50)
    classifier.fit(X, y, object_label='XOR task')
    yp = classifier.predict(X)
    print 'Predicted value is '
    print yp
    print 'Score for XOR problem is ', classifier.score(X, y)


def task_OR_problem(classifier):
    """
    Trains the classifier in the art of XOR problem
    :param classifier: any general classifier.
    :return: None
    """
    X = np.array([
                     [1, 0],
                     [0, 1],
                     [1, 1],
                     [0, 0]] * 50)
    y = np.array([1, 1, 1, 0] * 50)
    classifier.fit(X, y, object_label='OR task')
    yp = classifier.predict(X)
    print 'Predicted value is '
    print yp
    print 'Score for OR problem is ', classifier.score(X, y)


# Error can't do this, as this is multi class. Later will add support
def scikit_learn_dataset_training(classifier):
    from sklearn import datasets

    iris = datasets.load_iris()
    classifier.generic_task(iris.data, iris.target, 'Iris')


def class_digital_logic(classifier):
    """
    Trains in the art of 2 input, OR, and, xor.
    :param classifier:
    :return:
    """
    task_and(classifier)
    task_OR_problem(classifier)
    task_XOR_problem(classifier)




def simple_custom_fitting_class(classifier):
    """
    Fit with some simple custom functions.
    :param classifier: object RememberingVisualMachine
    :return:
    """
    # Lesson 1 11
    classifier.fit_custom_fx(dot_with_11, 2, 1, 'dot with [1,1]')
    # Lesson 2: Mean
    classifier.fit_custom_fx(meanie, 1500, 1, 'np.mean')


def caltech_101(classifier):
    root='/home/student/Downloads/101_ObjectCategories'
    categories = os.listdir(root)
    all_images = []
    for category_i in categories:
        all_images += glob.glob(root + '/' + category_i + '/*.jpg')
    from random import shuffle
    shuffle(all_images)
    elephant_list = glob.glob(root + '/' + categories[0] + '/*.jpg')
    negatives_list = all_images[:len(elephant_list)]
    x_total = elephant_list + negatives_list
    y = [1]*len(elephant_list) + [0]*len(negatives_list)
    x_train, x_test, y_train, y_test = train_test_split(x_total, y)
    classifier.fit(x_train, y_train, 'elephants')
    # TODO test this.