"""
Title: Remembering Machine

This classifier uses a pre-trained CNN as the front end and uses those
extracted features to form a persistent classifier that  remembers
from all the past classification task.

Update: 23 May 2015. Trying to make the classifiers growable, and not fixed width.
both height and width should be growable. Done
return label based on address out of classifier.
Next tasks: Generative model, given label, generate a sample

Update: 24 May 2015. Added a method for generating a sample from any label.
added methods to reflect, similar labels and daisy chain.

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
# Make sure that caffe is on the python path:
caffe_root = '/home/student/ln_onedrive/code/promising-patterns/caffe/'  # this file is expected to be in {caffe_root}/examples
sys.path.insert(0, caffe_root + 'python')
import caffe
import School
import types


# Classes
class ClassifierNode:
    """ A node that contains classifier, it's input address and output address.
    """

    def __init__(self, end_in_address, out_address, svm_c, classifier_name='Default',
                 given_predictor=None):
        self.out_address = out_address
        self.end_in_address = end_in_address  # end column
        self.label = classifier_name  # The name of this concept. e.g. like apple etc.
        # Check whether to create a standard classifier or a custom, given one.
        if given_predictor:
            self.given_predictor = given_predictor
            self.classifier_type = 'custom'
        else:
            self.classifier = svm.LinearSVC(C=svm_c, dual=False, penalty='l1')
            self.classifier_type = 'standard'

    def fit(self, x_in, y):
        """
        Fits the classifier and returns the score.
        :param x_in:
        :param y:
        :return: float, score of fitted classifier.
        """
        new_x_in = x_in[:, :self.end_in_address]
        self.classifier.fit(new_x_in, y)
        return self.classifier.score(new_x_in, y)

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
        return np.array([sigmoid(i) for i in dec_fx_in])
        # return dec_fx_in


class RememberingVisualMachine:
    """ A machine which stores both input X and the current output of bunch of classifiers.
    API should be similar to scikit learn"""

    def __init__(self, input_width, svm_c=1):
        """
        Initialize this class.

        :rtype : object self
        :param input_width: maximum input dimension. 4096 for caffe features.
        :param svm_c: the C for linear SVM
        :return: None
        """
        self.current_working_memory = np.zeros([1, input_width])
        self.prediction_column_start = input_width  # the start of classifiers output. Fixed.
        self.memory_width = input_width  # starting address for output for new classifier
        # also the width of the current working memory. Can grow.
        self.classifiers_list = []  # list of classifiers
        self.labels_list = []  # list of classifier names
        self.svm_c = svm_c

    def activate_working_memory(self, x_pred):
        """
        Given input, activate the previous predictors.
        :param: x_pred: input 2d matrix.
        :return: none
        """
        x_pred = np.array(x_pred, dtype=np.float)
        x_pred = normalize(x_pred)
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

    def fit(self, x_in, y, classifier_name='Default', relearn=False, save_classifier=False):
        """ Adds a new classifier and trains it, similar to Scikit API

        :param x_in: can either be a list of string, where strings are filepaths of images
                or they can be a 2d numpy matrix.
        :param predict_files_list: input files list, list of strings
        :param relearn: if given same classifier name, whether to relearn.
        :param save_classifier: should it write to memory the learnt classifier

        :returns: tuple of array and string.
            array is hard decision 1,0. String is the classifier class detected."""
        # check whether x_in is a files list or a numpy array
        if isinstance(x_in, types.ListType):
            # check for empty list
            if not len(x_in):
                print 'Hey man there is nothing in this task', classifier_name, ' returning.'
                return
            x_pred = np.vstack([copy.copy(extract_caffe_features(input_file))
                                for input_file in x_in])
        else:
            # x_in is numpy matrix
            x_pred = x_in
        self.fit_from_caffe_features(x_pred, y, classifier_name, relearn, save_classifier)

    def fit_from_caffe_features(self, x_in, y, classifier_name='Default', relearn=False,
                                save_classifier=True):
        """
        Adds a new classifier and trains it, similar to Scikit API

        :param x_in:np.array 2d, rows = no.of samples, columns=features
        :param y:  labels
        :param relearn: if given same classifier name, whether to relearn.
        :param save_classifier: should it write to memory the learnt classifier
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
        # activate all previous classifiers
        self.activate_working_memory(x_in)
        # Procure a new classifier, this might be wasteful, later perhaps reuse classifier
        # instead of lavishly getting new ones, chinese restaurant?
        new_classifier = ClassifierNode(end_in_address=self.memory_width,
                                        out_address=[self.memory_width],
                                        svm_c=self.svm_c,
                                        classifier_name=classifier_name)
        # Need to take care of mismatch in length of working memory and input samples.
        fit_score = new_classifier.fit(self.current_working_memory, y)
        print 'Fitting score for classifier ', classifier_name, ' is ', fit_score
        self.memory_width += 1
        # Update labels list
        self.classifiers_list.append(new_classifier)
        self.labels_list = [classifier_i.label for classifier_i in self.classifiers_list]
        if save_classifier:
            self.save(filename=classifier_file_name)  # caching of classifiers.

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

    def status(self, show_graph=False):
        """Gives out the current status, like number of classifier and prints their values

        :return: float, sparsity value"""

        print 'Currently there are ', len(self.classifiers_list), ' classifiers. They are'
        classifiers_coefficients = np.zeros([len(self.classifiers_list), self.memory_width])
        self.labels_list = [classifier_i.label for classifier_i in self.classifiers_list]
        print self.labels_list
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
            decimation_factor = int(self.current_working_memory.shape[0] / 40) + 1
            plt.figure()
            plt.imshow(self.current_working_memory[::decimation_factor,
                       self.prediction_column_start:
                       self.memory_width], interpolation='none', cmap='gray')
            plt.title('Current working memory')
            # Coefficients matrix plot
            plt.figure()
            plt.imshow(classifiers_coefficients, interpolation='none', cmap='gray')
            plt.title('Classifier coefficients')
            plt.figure()
            plt.imshow(classifiers_coefficients[:, self.prediction_column_start:], interpolation='none', cmap='gray')
            plt.title('Classifier interdependency')
            plt.show()

            # Coefficients matrix plot, sparsity, thresholded.
            plt.figure()
            classifiers_coefficients[classifiers_coefficients != 0] = 1
            # The mean number of non zero coefficients, 2 because its triangular.
            print 'Sparsity ratio is ', 2 * classifiers_coefficients.mean()
            plt.imshow(classifiers_coefficients, interpolation='none', cmap='gray')
            plt.title('Sparsity of coefficients')
            # classifier interdependency.
            plt.show()
            return 2 * classifiers_coefficients.mean()
        return 0

    def predict(self, x_in):
        """Give out what it thinks from the input.

        :param x_in: can either be a list of string, where strings are filepaths of images
                or they can be a 2d numpy matrix.
        :param: predict_files_list: input files list, list of strings

        :returns: tuple of array and string.
            array is hard decision 1,0. String is the classifier class detected."""
        # check whether x_in is a files list or a numpy array
        if os.path.isfile(x_in[0]):
            # x_in is a files list
            x_pred = np.vstack([copy.copy(extract_caffe_features(input_file))
                                for input_file in x_in])
        else:
            # x_in is numpy matrix
            x_pred = x_in
        return self.predict_from_features(x_pred)

    def predict_from_features(self, x_pred):
        """Give out what it thinks from the input. Input x_pred should be 2 dimensional.

        :param: x_pred: input files list, list of strings

        :returns: tuple of array and float.
            array is hard decision 1,0. float is the confidence."""
        input_number_samples, input_feature_dimension = x_pred.shape
        self.activate_working_memory(x_pred)
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
        return np.array(soft_dec > 0.5, dtype=np.int16), prediction_energy

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
        removing_index = len(self.labels_list) - self.labels_list[::-1].index(classifier_name)
        self.classifiers_list.pop(removing_index)
        print 'I no longer remember what a ', classifier_name, ' looks like :( '
        self.labels_list = [classifier_i.label for classifier_i in self.classifiers_list]
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
        for i, sample_i in enumerate(generated_samples):
            generated_matrix[i, :sample_i.shape[1]] = sample_i
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
        normalized_coefficients = coefficients / max_coefficient \
            if abs(max_coefficient) > 1e-4 else coefficients
        return normalized_coefficients.reshape(1, -1)

    def reflect(self, classifier_name, other_labels_length=10):
        """
        Given a label, generates it , an not it. And fits it again. Creates a new
        label called reflected_classifier_name
        :param classifier_name:
        :return:
        """
        print 'Reflecting on ', classifier_name
        generated_sample = self.generate_sample(classifier_name)
        positive_train = np.vstack([generated_sample] * other_labels_length)
        other_labels = [label_i for label_i in self.labels_list if label_i is not classifier_name]
        shuffle(other_labels)
        other_labels = other_labels[:other_labels_length]
        other_samples = self.generate_samples(other_labels)
        y = [1] * positive_train.shape[0] + [0] * other_samples.shape[0]
        # create a zeros matrix that is wide enough to hold both. Max of columns size
        X_train = np.zeros([len(y), max(other_samples.shape[1], positive_train.shape[1])])
        self.fit_from_caffe_features(X_train, y, 'reflected_' + classifier_name)

    def score(self, input_x, y):
        """
        Gives the accuracy between predicted( x_in) and y
        :param input_x: 2d matrix, samples x_in dimension
        :param y: actual label
        :return: float, between 0 to 1
        """
        # check whether input is file_list or 2d array
        if isinstance(input_x, types.ListType):
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

    def explain_interdependencies(self, max_tell=10):
        """ Using classifier in the list, explain which looks like what?
        """
        labels_list = self.labels_list
        classifiers_coefficients = np.zeros([len(self.classifiers_list), self.memory_width])
        for count, classifier_i in enumerate(self.classifiers_list):
            coeffs_i = classifier_i.classifier.coef_ \
                if classifier_i.classifier_type == 'standard' else np.zeros([1, 1])
            classifiers_coefficients[count, :coeffs_i.shape[1]] = coeffs_i
        interdependency_matrix = classifiers_coefficients[:, self.prediction_column_start:]
        print 'Now look at this interdependency matrix'
        plt.figure()
        plt.imshow(interdependency_matrix, interpolation='none', cmap='gray')
        plt.title('Interdependency matrix')
        plt.show()
        # get top 10 indices
        flattened_indices = np.argsort(interdependency_matrix, axis=None)[-10:]
        x_co_ords, y_co_ords = np.unravel_index(flattened_indices, interdependency_matrix.shape)
        for x_i, y_i in zip(x_co_ords, y_co_ords):
            print labels_list[x_i], ' looks like ', labels_list[y_i]
            # flattened_index = np.argmax(interdependency_matrix)
            # coords = np.unravel_index()
        # max_indices = np.argmax(interdependency_matrix, axis=1)
        # pick max_tell of max
        # for category_i in max_indices[:max_tell]:
        #     top_similarities = np.argmax(interdependency_matrix[category_i])
        #     sentence = 'The ' + labels_list[category_i] + ' looks like '
        #     valid = False
        #     for similar_i in top_similarities:
        #         if interdependency_matrix[category_i, similar_i] > 0.1:
        #             sentence += labels_list[similar_i]
        #             valid = True
        #     if valid:
        #         print sentence


# Global functions
# Reason for having 10 sigmoid is to get sharper distinction.
def sigmoid(x):
    return 1 / (1 + math.exp(-1 * x))


# Following are required for custom functions Task 1,2
def meanie(x):
    return np.mean(x, axis=1)


def dot_with_11(x):
    return np.dot(x, np.array([0.5, 0.5]))


def caffe_init():
    """
    Initialize the pre-trained Caffe classifier
    :return:
    the caffe net and transformer
    """
    if not os.path.isfile(caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'):
        print("Downloading pre-trained CaffeNet model...")
        from subprocess import call

        call(['../scripts/download_model_binary.py ../models/bvlc_reference_caffenet'])

    caffe.set_mode_cpu()
    net = caffe.Net(caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt',
                    caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel',
                    caffe.TEST)

    # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(
        1))  # mean pixel
    transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
    transformer.set_channel_swap('data', (2, 1, 0))  # the reference model has channels in BGR order instead of RGB
    return net, transformer


def extract_caffe_features(filename):
    """ Given a filename extracts the caffe features.
    Normalized by dividing by max.

    Argument:
        filename: string path to file
    """
    # cachine file feature.
    cached_caffe_file = filename[:-3] + 'caffe_feature.csv'
    if os.path.isfile(cached_caffe_file):
        feat = np.loadtxt(cached_caffe_file, delimiter=',')
    else:
        # Caching of caffe net. Initialize first time
        try:
            caffe_net
        except NameError:
            global caffe_net, caffe_transformer
            caffe_net, caffe_transformer = caffe_init()

        caffe_net.blobs['data'].data[...] = caffe_transformer.preprocess('data', caffe.io.load_image(filename))
        caffe_net.forward()
        feat = caffe_net.blobs['fc7'].data[0]
        np.savetxt(cached_caffe_file, feat, delimiter=',')
    max_feat = np.max(feat)
    normalized_feat = feat / max_feat if abs(max_feat) > 1e-4 else feat
    return normalized_feat


def caffe_directory(root_folder):
    """
    :param root_folder: path of root folder which contains subdirectory for each class.
    :return:
    """
    print 'Extracting caffe features from directory ', root_folder
    # get all the directories in the root folder.
    categories = [i for i in os.listdir(root_folder)
                  if os.path.isdir(os.path.join(root_folder, i))]
    # Hold one out teaching. For each category, that category is positive, rest are negative.
    for category_i in categories:
        caffe_file_name = root_folder + '/' + category_i + '.caffe_feature.csv'
        # caching, check if file exist.
        if not os.path.isfile(caffe_file_name):
            files_list = glob.glob(root_folder + '/' + category_i + '/*.jpg')
            files_list.extend(glob.glob(root_folder + '/' + category_i + '/*.JPEG'))
            caffe_matrix = np.vstack([copy.copy(extract_caffe_features(input_file))
                                      for input_file in files_list])
            np.savetxt(caffe_file_name, caffe_matrix, delimiter=',')
            print 'Features extracted from folder ', category_i, ' shape is ', caffe_matrix.shape
        else:
            print 'Caffe feature file exists for ', category_i


def caffinate_directory(root_folder):
    """
    call extract caffe on each file so that a cache is created for each file.
    :param root_folder: main directory.
    :return:
    """
    print 'Caffinating directory ', root_folder
    # get all the directories in the root folder.
    categories = [i for i in os.listdir(root_folder)
                  if os.path.isdir(os.path.join(root_folder, i))]
    # Hold one out teaching. For each category, that category is positive, rest are negative.
    for category_i in categories:
        files_list = glob.glob(root_folder + '/' + category_i + '/*.jpg')
        files_list.extend(glob.glob(root_folder + '/' + category_i + '/*.JPEG'))
        for input_file in files_list:
            extract_caffe_features(input_file)
    print 'Done caffinating.'

# Constants
input_dimension = 4096

if __name__ == '__main__':
    start_time = time.time()
    classifier_file_name = 'RememberingClassifier.pkl.gz'
    print 'Loading classifier file ...'
    if os.path.isfile(classifier_file_name):
        main_classifier = pickle.load(gzip.open(classifier_file_name, 'r'))
    else:
        main_classifier= RememberingVisualMachine(input_width=input_dimension, svm_c=1)
    print 'Loading complete.'
    # School.caltech_101(main_classifier)
    # School.caltech_101_test(main_classifier)
    # main_classifier.explain_interdependencies()
    School.cifar_school(main_classifier)
    main_classifier.status(show_graph=True)
    main_classifier.save(filename=classifier_file_name)
    print 'Total time taken to run this program is ', round((time.time() - start_time) / 60, ndigits=2), ' mins'

    # Scratch
    ##################################
    # Main_C1.remove_classifier('elephant')
    # caffe_directory(caltech101_root)
    # Main_C1.predict(elephant_files_list)
    # starfish_file = '/home/student/Downloads/101_ObjectCategories/starfish/image_0002.jpg'
    # starfish_caffe_feature = extract_caffe_features(starfish_file)
    # starfish_generated_sample = Main_C1.generate_sample('starfish').reshape(1,-1)
    # matrix_to_send_in = np.vstack([np.zeros([1, starfish_generated_sample.shape[1]]), starfish_generated_sample])
    # matrix_to_send_in[0, :starfish_caffe_feature.shape[0] ] = starfish_caffe_feature
    # Main_C1.predict_from_features(matrix_to_send_in)
    # Main_C1.predict_from_features(starfish_generated_sample.reshape(1,-1))
    # Main_C1.reflect('elephant')
    # caffinate_directory(paintings_root)
    # caltech101_root = '/home/student/Downloads/101_ObjectCategories'
    # paintings_root = '/home/student/Lpromising-patterns/paintings/data/two_class_full_size'
    # elephant_files_list = ['/home/student/Downloads/101_ObjectCategories/elephant/image_0002.jpg',
    #                        '/home/student/Downloads/101_ObjectCategories/elephant/image_0003.jpg',
    #                        '/home/student/Downloads/101_ObjectCategories/elephant/image_0004.jpg']
    # rhino_files_list = ['/home/student/Downloads/101_ObjectCategories/rhino/image_0002.jpg',
    #                     '/home/student/Downloads/101_ObjectCategories/rhino/image_0003.jpg',
    #                     '/home/student/Downloads/101_ObjectCategories/rhino/image_0004.jpg']
