"""
Title: Remembering Machine without feature extraction.
"""

__author__ = 'Abhishek Rao'

# Headers
import numpy as np
from sklearn import svm
import math
import matplotlib.pyplot as plt
import cPickle as pickle
from sklearn.metrics import f1_score
from sklearn.preprocessing import normalize
import sys
import os
import copy
import glob
import time
import gzip
from random import shuffle
import School


# Classes
class ClassifierNode:
    """ A node that contains classifier, it's input address and output address.
    """

    def __init__(self, end_in_address, out_address, svm_C, classifier_name='Default',
                 given_predictor=None):
        self.out_address = out_address
        self.end_in_address = end_in_address  # end column
        self.label = classifier_name  # The name of this concept. e.g. like apple etc.
        # Check whether to create a standard classifier or a custom, given one.
        if given_predictor:
            self.given_predictor = given_predictor
            self.classifier_type = 'custom'
        else:
            self.classifier = svm.LinearSVC(C=svm_C, dual=False, penalty='l1')
            self.classifier_type = 'standard'

    def fit(self, x_in, y):
        new_x_in = x_in[:, :self.end_in_address]
        self.classifier.fit(new_x_in, y)
        print 'Fitted with score = ', self.classifier.score(new_x_in, y)

    def predict(self, x_in):
        """
        Give output for the current classifier. Note instead of predict 1,0, better to use probability, soft prediction.
        :param x_in: The Classifier banks working memory, full matrix_in.
        :return: A column of predicted values.
        """
        new_x_in = x_in[:, :self.end_in_address]
        if self.classifier_type == 'standard':
            dec_fx_in = self.classifier.decision_function(new_x_in)
        else:
            dec_fx_in = self.given_predictor(new_x_in)
        # Convert it into mapping between 0 to 1 instead of -1 to 1
        # Sigmoid required? Think I'll remove it.
        return np.array([sigmoid_10(i) for i in dec_fx_in])
        # return dec_fx_in


class RememberingVisualMachine:
    """ A machine which stores both input X and the current output of bunch of classifiers.
    API should be similar to scikit learn"""

    def __init__(self, input_width, C=1):
        """
        Initialize this class.

        :rtype : object self
        :param input_width: maximum input dimension. 4096 for caffe features.
        :param front_end: a front end function that transforms the input to something else. e.g. a
            pre trained CNN. Any feature extractor.
        :param C: the SVM penalty term
        :return: None
        """
        self.current_working_memory = np.zeros([1, input_width])
        self.prediction_column_start = input_width  # the start of classifiers output. Fixed.
        self.memory_width = input_width  # starting address for output for new classifier
        # also the width of the current working memory. Can grow.
        self.classifiers_list = []  # list of classifiers
        self.labels_list = []  # list of classifier names
        self.param_svm_C = C


    def populate_working_memory(self, x_pred):
        """
        Given input, activate the previous predictors.
        :param: x_pred: input 2d matrix.
        :return: none
        """
        x_pred = np.array(x_pred,dtype=np.float)
        if len(x_pred.shape) is not 2:
            print "Error in predict. Input dimension should be 2"
            raise ValueError
        # x_pred = normalize(x_pred, axis=0) # normalize feature wise
        input_number_samples, input_feature_dimension = x_pred.shape
        # Create a blank slate for working with.
        self.current_working_memory = np.zeros([input_number_samples, self.memory_width])
        self.current_working_memory[:input_number_samples, :input_feature_dimension] = x_pred
        if len(self.classifiers_list) == 0:
            return
        for classifier_i in self.classifiers_list:
            predicted_value = classifier_i.predict(self.current_working_memory[:input_number_samples, :])
            predicted_shape = predicted_value.shape
            if len(predicted_shape) < 2:
                predicted_value = predicted_value.reshape(-1, 1)
            predicted_shape = predicted_value.shape
            self.current_working_memory[:predicted_shape[0], classifier_i.out_address] = predicted_value

    def predict(self, x_pred):
        """Give out what it thinks from the input. Input x_pred should be 2 dimensional.

        :param: x_pred: input 2d matrix.

        :returns: tuple of array and string.
            array is hard decision 1,0. String is the classifier class detected."""
        self.populate_working_memory(x_pred)
        input_number_samples, input_feature_dimension = x_pred.shape
        # Prediction scheme. Return the column in the classifier range (not input range) column with
        # highest variance. Note: memory width is next available address, so -1.
        prediction_range = self.current_working_memory[:input_number_samples,
                               self.prediction_column_start:self.memory_width]
        # Which column to choose? Now we are selecting column that has hightest sum.
        # Since decision function is signed distance from hyperplane, we want positives.
        # if we square we will get negatives too.
        prediction_energy = np.sum(prediction_range, axis=0)
        chosen_column = np.argmax(prediction_energy)
        soft_dec = prediction_range[:, chosen_column]
        print 'Looks like images of ', self.labels_list[chosen_column], ' confidence = ', \
            np.mean(np.square(soft_dec))
        # Do hard decision, return only 1,0
        return np.array(soft_dec > 0.5, dtype=np.int16),  prediction_energy

    def predict_last(self, x_pred):
        """
        Give the prediction value of latest classifier.

        :param x_pred: numpy 2d matrix, samples x features
        :return:
        """
        self.populate_working_memory(x_pred)
        return np.array(self.current_working_memory[:,-1] > 0.5, dtype=np.int16)


    def similar_labels(self, label):
        """
        Generates a sample from label class and then sees which are most similar.

        :param label:
        :return: 2 lists, first one labels list, second one confidence level for each.
        """
        generated_sample = self.generate_sample(label)
        _, prediction_energy = self.predict_from_features(generated_sample)
        # get the top ten active columns.
        sorted_indices = np.argsort(prediction_energy)[-10:][::-1]
        top_labels = np.array(self.labels_list)[sorted_indices]
        # print 'top labels similar to ', label, ' are ', top_labels
        # print 'their confidence levels are ', prediction_energy[sorted_indices]
        return top_labels, prediction_energy[sorted_indices]


    def daisy_chain(self, starting_label, chain_length=20):
        daisy_chain = [starting_label]
        for i in range(chain_length):
            returned_labels_list, _ = self.similar_labels(starting_label)
            starting_label = [i for i in returned_labels_list if i not in daisy_chain][0]
            daisy_chain.append(starting_label)
        return daisy_chain


    def fit(self, x_in, y, classifier_name='Default', relearn=False):
        """
        Adds a new classifier and trains it, similar to Scikit API

        :param x_in:np.array 2d, rows = no.of samples, columns=features
        :param y:  labels
        :return: None
        """
        # caching classifiers. Check if one has to relearn. if yes
        # the tries the score for this task. If score is good wont bother relearning.
        # else will relearn.
        if classifier_name in self.labels_list:
            print 'I have already been trained on the task of ', classifier_name
            if relearn:
                print 'Let me see how good I can remember this'
                score = self.score(x_in, y)
                if score > 0.5:
                    print 'I can do this task with F1 score of ', score, \
                        'so I wont bother learning again'
                    return
                else:
                    print 'Woah!O_o  this is totally different from what I know.', \
                        'will learn again'
            else:
                print 'I wont bother learning again. Feeling lazy :P '
                return
        # Normalize
        x_in = np.array(x_in, dtype=np.float)
        # x_in = normalize(x_in, axis=0) # normalize feature wise
        print 'Learning to recognize ', classifier_name, ' address will be ', self.memory_width
        input_number_samples, input_feature_dimension = x_in.shape
        if len(x_in.shape) is not 2:
            print "Error in predict. Input dimension should be 2"
            raise ValueError
        # activate previous classifiers
        self.populate_working_memory(x_in)
        # Procure a new classifier, this might be wasteful, later perhaps reuse classifier
        # instead of lavishly getting new ones, chinese restaurant?
        new_classifier = ClassifierNode(end_in_address=self.memory_width,
                                        out_address=[self.memory_width],
                                        svm_C= self.param_svm_C,
                                        classifier_name=classifier_name)
        # Need to take care of mismatch in length of working memory and input samples.
        new_classifier.fit(self.current_working_memory, y)
        self.memory_width += 1
        # Update labels list
        self.classifiers_list.append(new_classifier)
        self.labels_list = [classifier_i.label for classifier_i in self.classifiers_list]
        # self.save(filename=classifier_file_name)  # caching of classifiers.

    def fit_custom_fx(self, custom_function, input_width, output_width, task_name):
        """
        Push in a new custom function to classifiers list.
        :param custom_function: The function that will be used to predict. Should take in a 2D array input and
            give out a 2d array of same height and variable width.
        :param input_width: The width of input.
        :param output_width: The width of output. If a single neuron this is one.
        :param task_name: name of this function
        :return: None
        """
        new_classifier = ClassifierNode(
            end_in_address=input_width,
            out_address=self.memory_width + np.arange(output_width),
            classifier_name=task_name,
            given_predictor=custom_function
        )
        self.memory_width += output_width
        self.classifiers_list.append(new_classifier)

    def status(self, show_graph=False, list_classifier_name=True):
        """Gives out the current status, like number of classifier and prints their values"""
        print 'Currently there are ', len(self.classifiers_list), ' classifiers. They are'
        if list_classifier_name:
            print [i.label for i in self.classifiers_list]
        classifiers_coefficients = np.zeros([len(self.classifiers_list), self.memory_width])
        for count, classifier_i in enumerate(self.classifiers_list):
            coeffs_i = classifier_i.classifier.coef_ \
                if classifier_i.classifier_type == 'standard' else np.zeros([1, 1])
            classifiers_coefficients[count, :coeffs_i.shape[1]] = coeffs_i
            #    print 'Classifier: ', classifier_i
            #    print 'Classifier name: ', classifier_i.label
            #    print 'Out address', classifier_i.out_address
            #    print 'In address', classifier_i.end_in_address
            # print 'Coefficients: ', classifier_i.classifier.coef_, classifier_i.classifier.intercept_
        if show_graph:
            decimation_factor = int(self.current_working_memory.shape[0]/40)
            plt.imshow(self.current_working_memory[::decimation_factor,
                       self.prediction_column_start:
                self.memory_width], interpolation='none', cmap='gray')
            plt.title('Current working memory')
            plt.figure()
            plt.imshow(classifiers_coefficients, interpolation='none', cmap='gray')
            plt.title('Classifier coefficients')
            plt.show()

    def remove_classifier(self, classifier_name):
        """
        Removes the classifier whose name is same as classifier_name
        :param classifier_name: the label of the classifier to be removed.
        :return: the index of removed classifier. -1 if not found.
        """
        try:
            self.labels_list.index(classifier_name)
        except ValueError:
            print 'The specified label does not exist.'
            return -1
        # Get the last index
        removing_index = len(self.labels_list) - self.labels_list[::-1].index(classifier_name) -1
        self.classifiers_list.pop(removing_index)
        print 'I no longer remember what a ', classifier_name, ' looks like :( '
        self.labels_list = [classifier_i.label for classifier_i in self.classifiers_list]
        self.memory_width -= 1
        return removing_index


    def generate_samples(self, labels_list):
        """
        Creates samples of the given classifier_name in working memory.
        :param labels_list:  list of string, list of classifier names.
        :return: a 2d matrix of coefficients.
        """
        generated_samples = [self.generate_sample(label_i) for label_i in labels_list]
        widths = [i.shape[1] for i in generated_samples]
        generated_matrix = np.zeros([len(labels_list), max(widths)])
        for i,sample_i in enumerate(generated_samples):
            generated_matrix[i,:sample_i.shape[1]] = sample_i
        return generated_matrix


    def generate_sample(self, classifier_name):
        """
        Creates a sample of the given classifier_name and returns it.
        :param classifier_name: the label of the classifier to be generated.
        :return: the coefficients of classifier. as a 1 row, n_col matrix.
        """
        try:
            self.labels_list.index(classifier_name)
        except ValueError:
            print 'Dont know what the heck a ', classifier_name, ' is :( . Sorry.'
            raise ValueError
        classifier_index = self.labels_list.index(classifier_name)
        coefficients = self.classifiers_list[classifier_index].classifier.coef_
        print 'Generating a sample of ', classifier_name
        max_coefficient = np.max(coefficients)
        normalized_coefficients = coefficients/max_coefficient \
            if abs(max_coefficient) > 1e-4 else coefficients
        return normalized_coefficients.reshape(1,-1)


    def reflect(self, classifier_name, other_labels_length = 10):
        """
        Given a label, generates it , an not it. And fits it again. Creates a new
        label called reflected_classifier_name
        :param classifier_name:
        :return:
        """
        print 'Reflecting on ', classifier_name
        generated_sample = self.generate_sample(classifier_name)
        positive_train = np.vstack([generated_sample]*other_labels_length)
        other_labels = [label_i for label_i in self.labels_list if label_i is not classifier_name]
        shuffle(other_labels)
        other_labels = other_labels[:other_labels_length]
        other_samples = self.generate_samples(other_labels)
        y = [1]*positive_train.shape[0] + [0]*other_samples.shape[0]
        # create a zeros matrix that is wide enough to hold both. Max of columns size
        X_train = np.zeros([len(y), max(other_samples.shape[1], positive_train.shape[1])])
        self.fit_from_caffe_features(X_train, y, 'reflected_'+classifier_name)


    def score(self, input_x, y):
        """
        Gives the accuracy between predicted( x_in) and y
        :param input_x: 2d matrix, samples x_in dimension
        :param y: actual label
        :return: float, between 0 to 1
        """
        # check whether input is file_list or 2d array
        if input_x[0] is str:
            yp_score, _ = self.predict(input_x)
            return f1_score(y, y_pred=yp_score)
        else:
            yp_score, _ = self.predict_from_features(input_x)
            return f1_score(y, y_pred=yp_score)


    def generic_task(self, x_in, y, task_name):
        """
        A generic framework to train on different tasks.
        """
        self.fit(x_in, y, classifier_name=task_name)
        print 'The score for task ', task_name, ' is ', self.score(x_in, y)

    def save(self, filename="RememberingClassifier.pkl"):
        """
        Pickle thyself.
        """
        pickle.dump(self, gzip.open(filename, 'w'))
        print 'Remembering Machine saved.'

    def remove_duplicates(self):
        print 'removing duplicates'
        duplicates_list = []
        for label_i in self.labels_list:
            if self.labels_list.count(label_i) > 1:
                if label_i not in duplicates_list:
                    duplicates_list.append(label_i)
        if duplicates_list is not []:
            print 'Found duplicates ', duplicates_list
            for label_i in duplicates_list:
                self.remove_classifier(label_i)
        else:
            print 'No duplicates found.'

    def visualize_clf(self, X, y):
        """
        Draw a 2d boundary for first two features for the last classifier.
        :return: none
        """
        clf = self.classifiers_list[-1].classifier

        # create a mesh to plot in
        h = 0.2 # step size of mesh
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        for i, clf in enumerate([self]):
            # Plot the decision boundary. For that, we will assign a color to each
            # point in the mesh [x_min, m_max]x[y_min, y_max].
            # plt.subplot(2, 2, i + 1)
            # plt.subplots_adjust(wspace=0.4, hspace=0.4)

            Z = clf.predict_last(np.c_[xx.ravel(), yy.ravel()])

            # Put the result into a color plot
            Z = Z.reshape(xx.shape)
            plt.contourf(xx, yy, Z, cmap='gray', alpha=0.8)

            # Plot also the training points
            plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
            plt.xlim(xx.min(), xx.max())
            plt.ylim(yy.min(), yy.max())
            plt.title('Classifier ' + str(len(self.classifiers_list)))
        plt.show()

# Global functions
# Reason for having 10 sigmoid is to get sharper distinction.
def sigmoid_10(x):
    return 1 / (1 + math.exp(-10 * x))


# Following are required for custom functions Task 1,2
def meanie(x):
    return np.mean(x, axis=1)


def dot_with_11(x):
    return np.dot(x, np.array([0.5, 0.5]))
# Constants
input_dimension = 2


if __name__ == '__main__':
    start_time = time.time()
    learning_phase = False
    classifier_file_name = 'RememberingClassifier_2.pkl.gz'
    print 'Loading classifier file ...'
    if os.path.isfile(classifier_file_name):
        Main_C1 = pickle.load(gzip.open(classifier_file_name, 'r'))
    else:
        Main_C1 = RememberingVisualMachine(input_width=input_dimension)
    print 'Loading complete.'

    # Main_C1.remove_classifier('task of far points and 4,4')
    # School.train_tri_band(Main_C1)
    School.random_linear_trainer(Main_C1, stages=20)
    School.random_linear_trainer2(Main_C1, stages=20)
    Main_C1.status(show_graph=True)
    School.growing_complex_trainer(Main_C1)
    # Main_C1.visualize_clf(x_total, y)
    Main_C1.status(show_graph=True)
    # Main_C1.save(filename=classifier_file_name)
    print 'Total time taken to run this program is ', round((time.time() - start_time)/60, ndigits=2), ' mins'

