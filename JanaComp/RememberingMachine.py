"""
Title: Remembering Machine

This classifier uses a pre-trained CNN as the front end and uses those
extracted features to form a persistent classifier that  remembers
from all the past classification task.

Update: 23 May 2015. Trying to make the classifiers growable, and not fixed width.
both height and width should be growable. Done
return label based on address out of classifier.
"""

__author__ = 'Abhishek Rao'

# Headers
import numpy as np
from sklearn import svm
import math
import matplotlib.pyplot as plt
import cPickle as pickle
from sklearn.metrics import f1_score
import sys
import os
import copy
import glob
import time
# Make sure that caffe is on the python path:
caffe_root = '/home/student/ln_onedrive/code/promising-patterns/caffe/'  # this file is expected to be in {caffe_root}/examples
sys.path.insert(0, caffe_root + 'python')
import caffe
import School


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
        # Sigmoid required? Think I'll remove it.
        # return np.array([sigmoid_10(i) for i in dec_fx_in])
        return dec_fx_in


class RememberingVisualMachine:
    """ A machine which stores both input X and the current output of bunch of classifiers.
    API should be similar to scikit learn"""

    def __init__(self, input_width):
        """
        Initialize this class.

        :rtype : object self
        :param input_width: maximum input dimension. 4096 for caffe features.
        :param front_end: a front end function that transforms the input to something else. e.g. a
            pre trained CNN. Any feature extractor.
        :return: None
        """
        self.current_working_memory = np.zeros([1, input_width])
        self.prediction_column_start = input_width  # the start of classifiers output. Fixed.
        self.memory_width = input_width  # starting address for output for new classifier
        # also the width of the current working memory. Can grow.
        self.classifiers_list = []

    def predict(self, predict_files_list):
        """Give out what it thinks from the input. Input x_pred should be 2 dimensional.

        :param: predict_files_list: input files list, list of strings

        :returns: tuple of array and string.
            array is hard decision 1,0. String is the classifier class detected."""
        x_pred = np.vstack([copy.copy(extract_caffe_features(input_file))
                            for input_file in predict_files_list])
        input_number_samples, input_feature_dimension = x_pred.shape
        # Create a blank slate for working with.
        self.current_working_memory = np.zeros([input_number_samples, self.memory_width])
        if len(x_pred.shape) is not 2:
            print "Error in predict. Input dimension should be 2"
            raise ValueError
        self.current_working_memory[:input_number_samples, :input_feature_dimension] = x_pred
        for classifier_i in self.classifiers_list:
            predicted_value = classifier_i.predict(self.current_working_memory[:input_number_samples, :])
            predicted_shape = predicted_value.shape
            if len(predicted_shape) < 2:
                predicted_value = predicted_value.reshape(-1, 1)
            predicted_shape = predicted_value.shape
            self.current_working_memory[:predicted_shape[0], classifier_i.out_address] = predicted_value
        # Prediction scheme. Return the column in the classifier range (not input range) column with
        # highest variance.
        prediction_range = self.current_working_memory[:input_number_samples,
                               self.prediction_column_start:self.memory_width]
        # Which column to choose? Now we are selecting column that has hightest sum.
        # Since decision function is signed distance from hyperplane, we want positives.
        # if we square we will get negatives too.
        prediction_energy = np.sum(prediction_range, axis=0)
        chosen_column = np.argmax(prediction_energy)
        classifier_labels = [classifier_i.label for classifier_i
                             in self.classifiers_list]  # assuming single width
        soft_dec = prediction_range[:, chosen_column]
        print 'Looks like images of ', classifier_labels[chosen_column], ' confidence = ', \
            np.mean(np.square(soft_dec))
        # Do hard decision, return only 1,0
        return np.array(soft_dec > 0, dtype=np.int16), classifier_labels[chosen_column],


    def fit(self, input_file_list, y, object_label='Default', relearn=False):
        """
        Adds a new classifier and trains it, similar to Scikit API

        :type input_file_list: list
        :param input_file_list: Input image files list
        :param y:  labels
        :param relearn: boolean, default False, if True if the label exists,
            predicts and checks score. If it is low, relearns the task.
        :return: None
        """
        # caching classifiers. Check if one has to relearn. if yes
        # the tries the score for this task. If score is good wont bother relearning.
        # else will relearn.
        if object_label in [classifier_i.label for classifier_i in self.classifiers_list]:
            print 'I have already been trained on the task of ', object_label
            if relearn:
                print 'Let me see how good I can remember this'
                score = self.score(input_file_list, y)
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
        print 'Learning to recognize ', object_label, ' address will be ', self.memory_width
        x_in = np.vstack([copy.copy(extract_caffe_features(input_file))
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
        self.current_working_memory = np.zeros([input_number_samples, self.memory_width])
        self.current_working_memory[:, :x_in.shape[1]] = x_in
        # Procure a new classifier, this might be wasteful, later perhaps reuse classifier
        # instead of lavishly getting new ones, chinese restaurant?
        new_classifier = ClassifierNode(end_in_address=self.memory_width,
                                        out_address=[self.memory_width],
                                        classifier_name=object_label)
        # Need to take care of mismatch in length of working memory and input samples.
        new_classifier.fit(self.current_working_memory, y)
        self.memory_width += 1
        self.classifiers_list.append(new_classifier)
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
        """Gives out the current status, like number of classifier and prints their values"""
        print 'Currently there are ', len(self.classifiers_list), ' classifiers. They are'
        classifiers_coefficients = np.zeros([len(self.classifiers_list), self.memory_width])
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
            plt.imshow(self.current_working_memory[:200, self.prediction_column_start:
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
        labels_list = [classifier_i.label for classifier_i in self.classifiers_list]
        try:
            labels_list.index(classifier_name)
        except ValueError:
            print 'The specified label does not exist.'
            return -1
        removing_index = labels_list.index(classifier_name)
        self.classifiers_list.pop(removing_index)
        print 'I no longer remember what a ', classifier_name, ' looks like :( '
        return removing_index

    def score(self, files_list, y):
        """
        Gives the accuracy between predicted( x_in) and y
        :param files_list: 2d matrix, samples x_in dimension
        :param y: actual label
        :return: float, between 0 to 1
        """
        yp_score, _ = self.predict(files_list)
        return f1_score(y, y_pred=yp_score)

    def generic_task(self, x_in, y, task_name):
        """
        A generic framework to train on different tasks.
        """
        self.fit(x_in, y, object_label=task_name)
        print 'The score for task ', task_name, ' is ', self.score(x_in, y)

    def save(self, filename="RememberingClassifier.pkl"):
        """
        Pickle thyself.
        """
        pickle.dump(self, open(filename, 'w'))
        print 'Remembering Machine saved.'


# Global functions
# Reason for having 10 sigmoid is to get sharper distinction.
def sigmoid_10(x):
    return 1 / (1 + math.exp(-10 * x))


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

    Argument:
        filename: string path to file
    """
    # cachine file feature.
    cached_caffe_file = filename[:-3] + 'caffe_feature.csv'
    if os.path.isfile(cached_caffe_file):
        return np.loadtxt(cached_caffe_file, delimiter=',')
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
    return feat


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
        caffe_file_name = root_folder+ '/' + category_i + '.caffe_feature.csv'
        # caching, check if file exist.
        if not os.path.isfile(caffe_file_name):
            files_list = glob.glob(root_folder + '/' + category_i + '/*.jpg')
            caffe_matrix = np.vstack([copy.copy(extract_caffe_features(input_file))
                                for input_file in files_list])
            np.savetxt(caffe_file_name, caffe_matrix, delimiter=',')
            print 'Features extracted from folder ', category_i, ' shape is ', caffe_matrix.shape
        else:
            print 'Caffe feature file exists for ', category_i


# Constants
input_dimension = 4096


if __name__ == '__main__':
    start_time = time.time()
    learning_phase = True
    classifier_file_name = 'RememberingClassifier.pkl'
    caltech101_root = '/home/student/Downloads/101_ObjectCategories'
    # rhino_files_list = ['/home/student/Downloads/101_ObjectCategories/rhino/image_0002.jpg',
    #                     '/home/student/Downloads/101_ObjectCategories/rhino/image_0003.jpg',
    #                     '/home/student/Downloads/101_ObjectCategories/rhino/image_0004.jpg']
    if os.path.isfile(classifier_file_name):
        Main_C1 = pickle.load(open(classifier_file_name, 'r'))
    else:
        Main_C1 = RememberingVisualMachine(input_width=input_dimension)
    if learning_phase:
        School.caltech_101(Main_C1)
        School.caltech_101_test(Main_C1, max_categories=2)
    # Main_C1.remove_classifier('elephant')
    # caffe_directory(caltech101_root)
    # Main_C1.predict(rhino_files_list)
    Main_C1.status(show_graph=True)
    Main_C1.save(filename=classifier_file_name)
    print 'Total time taken to run this program is ', round((time.time() - start_time)/60, ndigits=2), ' mins'
