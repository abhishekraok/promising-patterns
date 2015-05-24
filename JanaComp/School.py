""" This is the place where young remembering machine learning classifiers are trained in various arts.
"""
__author__ = 'Abhishek Rao'

# Headers
import numpy as np
from LearningMachines2 import meanie, dot_with_11
from sklearn.cross_validation import train_test_split
import os
import glob
from random import shuffle


def task_and(classifier):
    # Task 1 noisy and
    noise_columns = np.random.randn(90, 3)
    data_columns = np.array([[1, 0], [0, 1], [1, 1], [0, 0]] * 2 + [[0, 1]])
    data_columns_big = np.vstack([data_columns] * 10)
    X = np.hstack([data_columns_big, noise_columns])  # total data
    print X
    y = np.array([[0, 0, 1, 0] * 2 + [0]])
    y_big = np.hstack([y] * 10).flatten()
    classifier.fit(X, y_big, classifier_name='Noisy and long')
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
    classifier.fit(X, y, classifier_name='XOR task')
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
    X = np.array([[1, 0],
                  [0, 1],
                  [1, 1],
                  [0, 0]] * 50)
    y = np.array([1, 1, 1, 0] * 50)
    classifier.fit(X, y, classifier_name='OR task')
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


def caltech_101(classifier, negatives_samples_ratio=2, max_categories=None):
    print 'CalTech 101 dataset training started'
    root = '/home/student/Downloads/101_ObjectCategories'
    categories = [i for i in os.listdir(root)
                  if os.path.isdir(os.path.join(root, i))]
    small_categories = categories[:max_categories] if max_categories is not None else categories
    # Hold one out teaching. For each category, that category is positive, rest are negative.
    for category_i in small_categories:
        positive_list = glob.glob(root + '/' + category_i + '/*.jpg')
        negatives_list = []
        for other_category_i in categories:
            if other_category_i != category_i:
                negatives_list += glob.glob(root + '/' + other_category_i + '/*.jpg')
        shuffle(negatives_list)
        positive_samples_count = len(positive_list)
        small_negative_list = negatives_list[:positive_samples_count * negatives_samples_ratio]
        x_total = positive_list + small_negative_list
        y = [1]*len(positive_list) + [0]*len(small_negative_list)
        x_train, x_test, y_train, y_test = train_test_split(x_total, y)
        task_name = 'CalTech101_' + category_i
        classifier.fit(x_train, y_train, task_name)

def caltech_101_test(classifier, max_categories=None):
    """
    A test to see how well CalTech 101 was learnt.

    :param classifier:
    :return: The mean F1 score.
    """
    print 'Exam time! time for the CalTech101 test. All the best'
    root = '/home/student/Downloads/101_ObjectCategories'
    categories = [i for i in os.listdir(root)
                  if os.path.isdir(os.path.join(root, i))]
    small_categories = categories[:max_categories] if max_categories is not None else categories
    # Hold one out teaching. For each category, that category is positive, rest are negative.
    score_sheet = []  # Place to store all the scores.
    for category_i in small_categories:
        positive_list = glob.glob(root + '/' + category_i + '/*.jpg')
        negatives_list = []
        for other_category_i in categories:
            if other_category_i != category_i:
                negatives_list += glob.glob(root + '/' + other_category_i + '/*.jpg')
        shuffle(negatives_list)
        positive_samples_count = len(positive_list)
        small_negative_list = negatives_list[:positive_samples_count *2]
        x_total = positive_list + small_negative_list
        y = [1]*len(positive_list) + [0]*len(small_negative_list)
        x_train, x_test, y_train, y_test = train_test_split(x_total, y)
        score = classifier.score(x_test, y_test)
        print 'In the category ', category_i, ' F1 score is ', score
        score_sheet.append(score)
    print 'The mean F1 score among all the classes is ', np.mean(score_sheet)
    return np.mean(score_sheet)

