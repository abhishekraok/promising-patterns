"""
Title: Remembering Machine

This classifier uses a pre-trained CNN as the front end and uses those
extracted features to form a persistent classifier that  remembers
from all the past classification task.

"""

# TODO  Use this as the input for remembering classifier.

__author__ = 'Abhishek Rao'

# Headers
import numpy as np
from sklearn import svm
import math
import matplotlib.pyplot as plt
import pickle
import os.path
from sklearn.metrics import f1_score
from sklearn.cross_validation import train_test_split
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
import sys
import os
from glob import glob
from sklearn.metrics import average_precision_score, classification_report
import Image
import copy
from random import shuffle
import copy

# Make sure that caffe is on the python path:
caffe_root = '/home/student/ln_onedrive/code/promising-patterns/caffe/'  # this file is expected to be in {caffe_root}/examples

sys.path.insert(0, caffe_root + 'python')
import caffe
import School


# Constants


# Classes
class ClassifierNode:
    """ A node that contains classifier, it's input address and output address.
    """

    def __init__(self, end_in_address, out_address, classifier_name='Default',
                 given_predictor=None):
        self.out_address = out_address
        self.end_in_address = end_in_address  # end column
        self.label = classifier_name  # The name of this concept. e.g. like apple etc.
        # Check whether to create a standard classifier or a custom, given one.
        if given_predictor:
            self.given_predictor = given_predictor
            self.classifier_type = 'custom'
        else:
            self.classifier = svm.LinearSVC(dual=False, penalty='l1')
            self.classifier_type = 'standard'

    def fit(self, x_in, y):
        new_x_in = x_in[:, :self.end_in_address]
        self.classifier.fit(new_x_in, y)

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
        return np.array([sigmoid_10(i) for i in dec_fx_in])


class RememberingVisualMachine:
    """ A machine which stores both input X and the current output of bunch of classifiers.
    API should be similar to scikit learn"""

    def __init__(self, max_width, input_width, height, front_end=None):
        """
        Initialize this class.

        :rtype : object self
        :param max_width: maximum data dimension in current working memory, should be greater than
            input_width.
        :param input_width: maximum input dimension.
        :param height: maximum number of input samples
        :param front_end: a front end function that transforms the input to something else. e.g. a
            pre trained CNN. Any feature extractor.
        :return: None
        """
        self.current_working_memory = np.zeros([height, max_width])
        self.prediction_column_start = input_width  # the start of classifiers output.
        self.classifiers_current_count = 0  # starting address for output for new classifier
        self.classifiers_list = []

    def predict(self, predict_files_list):
        """Give out what it thinks from the input. Input x_pred should be 2 dimensional.

        :param: predict_files_list: input files list, list of strings"""
        self.current_working_memory *= 0  # Flush the current input
        x_pred = np.vstack([copy.copy(extract_caffe_features(caffe_net, caffe_transformer, input_file))
                          for input_file in predict_files_list])
        input_number_samples, input_feature_dimension = x_pred.shape
        if len(x_pred.shape) is not 2:
            print "Error in predict. Input dimension should be 2"
            raise ValueError
        self.current_working_memory[:input_number_samples, :input_feature_dimension] = x_pred
        for classifier_i in self.classifiers_list:
            predicted_value = classifier_i.predict(self.current_working_memory)
            predicted_shape = predicted_value.shape
            if len(predicted_shape) < 2:
                predicted_value = predicted_value.reshape(-1, 1)
            predicted_shape = predicted_value.shape
            self.current_working_memory[:predicted_shape[0], classifier_i.out_address] = predicted_value
        # Prediction scheme. Return the column in the classifier range (not input range) column with
        # highest variance.
        prediction_range = self.current_working_memory[:input_number_samples,
                           self.prediction_column_start:self.prediction_column_start \
                                                            + self.classifiers_current_count]
        prediction_variances = np.var(prediction_range, axis=0)  # column wise variance
        chosen_column = np.argmax(prediction_variances)
        soft_dec = prediction_range[:, chosen_column]
        return np.array(soft_dec > 0.5, dtype=np.int16)

    def fit(self, input_file_list, y, object_label='Default'):
        """
        Adds a new classifier and trains it, similar to Scikit API

        :type input_file_list: list
        :param input_file_list: Input image files list
        :param y:  labels
        :return: None
        """
        # check for limit reach for number of classifiers.
        if self.classifiers_current_count + self.prediction_column_start \
                > self.current_working_memory.shape[1]:
            print 'No more space for classifier. ERROR'
            raise MemoryError

        x_in = np.vstack([copy.copy(extract_caffe_features(caffe_net, caffe_transformer, input_file))
                          for input_file in input_file_list])
        self.fit_from_caffe_features(x_in, y, object_label)

    def fit_from_caffe_features(self, x_in, y, object_label='Default'):
        """
        Adds a new classifier and trains it, similar to Scikit API

        :param x_in:np.array 2d, rows = no.of samples, columns=features
        :param y:  labels
        :return: None
        """
        input_number_samples, input_feature_dimension = x_in.shape
        if len(x_in.shape) is not 2:
            print "Error in predict. Input dimension should be 2"
            raise ValueError
        self.current_working_memory[:x_in.shape[0], :x_in.shape[1]] = x_in
        # Procure a new classifier, this might be wasteful, later perhaps reuse classifier
        # instead of lavishly getting new ones, chinese restaurant?
        new_classifier = ClassifierNode(
            end_in_address=self.prediction_column_start + self.classifiers_current_count,
            out_address=[self.prediction_column_start + self.classifiers_current_count + 1],
            classifier_name=object_label)
        self.classifiers_current_count += 1
        # Need to take care of mismatch in length of working memory and input samples.
        new_classifier.fit(self.current_working_memory[:input_number_samples], y)
        self.classifiers_list.append(new_classifier)

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
            out_address=self.prediction_column_start + self.classifiers_current_count + np.arange(output_width),
            classifier_name=task_name,
            given_predictor=custom_function
        )
        self.classifiers_current_count += output_width
        self.classifiers_list.append(new_classifier)

    def status(self, show_graph=False):
        """Gives out the current status, like number of classifier and prints their values"""
        print 'Currently there are ', len(self.classifiers_list), ' classifiers. They are'
        classifiers_coefficients = np.zeros(self.current_working_memory.shape)
        print [classifier_i.label for classifier_i in self.classifiers_list]
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
            plt.imshow(self.current_working_memory[:200,4096:4196], interpolation='none', cmap='gray')
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
            labels_list = [classifier_i.label for classifier_i in self.classifiers_list]
        except ValueError:
            print 'The specified label does not exist.'
            return -1
        removing_index = labels_list.index(classifier_name)
        self.classifiers_list.pop(removing_index)
        print 'Classifier was removed. Its nae was', classifier_name
        return removing_index

    def score(self, x_in, y):
        """
        Gives the accuracy between predicted( x_in) and y
        :param x_in: 2d matrix, samples x_in dimension
        :param y: actual label
        :return: float, between 0 to 1
        """
        yp_score = self.predict(x_in)
        return f1_score(y, y_pred=yp_score)

    def generic_task(self, x_in, y, task_name):
        """
        A generic framework to train on different tasks.
        """
        self.fit(x_in, y, object_label=task_name)
        print 'The score for task ', task_name, ' is ', self.score(x_in, y)


# Global functions
# Reason for having 10 sigmoid is to get sharper distinction.
def sigmoid_10(x):
    return 1 / (1 + math.exp(-10*x))


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
    # plt.rcParams['figure.figsize'] = (10, 10)
    # plt.rcParams['image.interpolation'] = 'nearest'
    # plt.rcParams['image.cmap'] = 'gray'

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
    transformer.set_transpose('data', (2,0,1))
    transformer.set_mean('data', np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1)) # mean pixel
    transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
    transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
    return net, transformer

def extract_caffe_features(in_net, transformer_e, filename):
    """ Given a filename extracts the caffe features.

    Argument:
        in_net: The caffe net
        filename: string path to file
    """
    in_net.blobs['data'].data[...] = transformer_e.preprocess('data', caffe.io.load_image(filename))
    in_net.forward()
    feat = in_net.blobs['fc7'].data[0]
    return feat

if __name__ == '__main__':
    learning_phase = False
    classifier_file_name = 'RememberingClassifier.pkl'
    caffe_net, caffe_transformer = caffe_init()
    if os.path.isfile(classifier_file_name):
        Main_C1 = pickle.load(open(classifier_file_name, 'r'))
    else:
        Main_C1 = RememberingVisualMachine(max_width=8000, input_width=4096, height=1000)

    # Main_C1.fit(small_file_list, y, 'cat')
    # Learn or not learn?
    # if learning_phase:
    #     School.class_digital_logic(Main_C1)
    #     School.simple_custom_fitting_class(Main_C1)
    # Main_C1.fit_custom_fx(np.mean,input_width=1500, output_width=1, task_name='np.mean')
    # yp = Main_C1.predict(np.random.randn(8, 22))
    # print 'Predicted value is ', yp
    # Main_C1.remove_classifier('np.mean')
    School.caltech_101(Main_C1)
    School.caltech_101_test(Main_C1)
    Main_C1.status(show_graph=False)
    pickle.dump(Main_C1, open(classifier_file_name, 'w'))

    # Scratch -----------------------------------------------------
    # if os.path.isfile(cat_file):
    #     cat_features = copy.copy(extract_caffe_features(caffe_net, transformer, cat_file))
    #     fish_features = copy.copy(extract_caffe_features(caffe_net, transformer, fish_file))
    #     print 'caffe features extracted.'
    #     plt.plot(cat_features, '.')
    #     plt.plot(fish_features, 'r.')
    #     plt.plot
    #     plt.show()
