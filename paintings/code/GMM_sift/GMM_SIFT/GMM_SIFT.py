""" File to classify images based on GMM and SIFT.
Author : Abhishek Rao
Date: 26 Feb 2015
"""
import numpy as np
from sklearn import mixture
import skimage.io as io
import matplotlib.pyplot as plt
import time
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
import cPickle
import os
import uuid
import cv2
from random import shuffle
import progressbar
from time import sleep
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt
# from sklearn import cross_validation


# ########### Constants ################
# TODO Change this while testing increase
GMM_COMPONENTS = 32  # Number of GMM components in Mixture modeling.
GMM_CONVERGENCE_THRESHOLD = 0.01  # GMM convergence, when to stop fitting
MAX_IMAGE_PER_CLASS = 10  # Maximum images to read per class, limits
# too big data set from being read
UNIQUE_RUN_NAME = str(uuid.uuid4())


# ########### Classes ###################
class GmmImageSIFT():
    """Class where objects have image, label and GMM"""

    def __init__(self, image=None):
        """can initialize with an image"""
        if image is not None:
            self.image = image
        else:
            self.image = image
        self.label = None  # which class they belong
        self.predicted_label = None  # for test image only, predict label
        self.sift_des = None  # Store the SIFT descriptors, matrix of n-rows
        # and 128 columns
        self.gmm_model = None  # GMM model that will be trained

    def fit(self, order=None):
        """create and fit GMM to self.
        Find SIFT descriptor for current image, fit that into GMM"""
        self.sift_des = get_sift_des(self.image)
        order_used = GMM_COMPONENTS if order is None else order
        self.gmm_model = mixture.GMM(n_components=order_used,
                                     thresh=GMM_CONVERGENCE_THRESHOLD)
        # If the number of sift keypoints is less than order then increase
        # it
        while self.sift_des.shape[0] < order_used:
            self.sift_des = np.vstack([self.sift_des, self.sift_des])
        self.gmm_model.fit(self.sift_des)
        return self

    def score(self, x):
        """Given an object of class GmmImageSIFT X find the sum log prob with
        current object
        :X: object of class GmmImageSIFT

        Returns:
            the fit score of current model with input X image
        """
        sift_des = get_sift_des(x.image)
        return sum(self.gmm_model.score(sift_des))


# ########### functions ######################
def get_sift_des(image):
    """ Given a RGB or grayscale image, returns a matrix where the rows are
    sift descriptors and 128 columns are present."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    sift = cv2.SIFT()
    des = sift.detectAndCompute(gray, None)[1]
    # some images are too simple, no descriptor will be found, so create a
    # empty dummy descriptor to prevent it from being null
    if des is None:
        des = np.zeros((1,128))
    return des


def visualize3d(input3darray, color='b'):
    """Scatter plot 3d array"""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(input3darray[:, 0], input3darray[:, 1], input3darray[:, 2], c=color)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.draw()
    time.sleep(0.1)


def full_image_vis(image):
    """Given an image of (x,y,rgb), displays it, shows pixel scatter plot
    and then the gmm fit of it."""
    fig = plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(image)
    plt.title('The image')
    reshaped_image = np.vstack([image[:, :, k].ravel() for k in range(3)]).T
    plt.subplot(2, 2, 2)
    ax = fig.add_subplot(222, projection='3d')
    ax.scatter(reshaped_image[::20, 0], reshaped_image[::20, 1], reshaped_image[::20, 2])
    plt.title('The RGB values scatter plot')
    g = mixture.GMM(n_components=50)
    print 'fitting ...'
    g.fit(reshaped_image)
    print 'fitting done'
    ax = fig.add_subplot(223, projection='3d')
    ax.scatter(g.means_[:, 0], g.means_[:, 1], g.means_[:, 2], c='r')
    plt.title('The GMM means of the RGB scatter plot')
    plt.show()


def model_order_selection(input_train_set):
    """
    Given an GmmImageSIFT collection input_train_set, reshape it and fit a GMM for
    it, check it's bic. Pick the lowest. Return integer.
    We will choose randomly 10 out of this and average over it.

    Arguments:
        input_train_set: list of objects of class GmmImageSIFT.

    Return:
        Integer, highest bic score order
    """
    print 'Finding best model order by bic'
    orders = [2, 4, 8, 16, 32]
    total_selection = 11  # pick this many out of total to average
    shuffle(input_train_set)  # Shuffle data
    set_to_use = input_train_set[:total_selection]  # choose few
    image_bic_scores = []
    for X in set_to_use:
        x_sif_des = get_sift_des(X.image)
        # Check to make sure the number of descriptors, i.e. rows > order
        bic_scores_orderwise = [(X.fit(order=i)).gmm_model.bic(x_sif_des)
                                for i in orders if x_sif_des.shape[0] > i]
        image_bic_scores.append(bic_scores_orderwise)
    # Average by image, need to average by column
    # Each image will be a row, each column an order

    average_order = np.mean(np.array(image_bic_scores), axis=0)
    best_order = orders[average_order.argmin()]
    print 'The bic scores are ', average_order, ' best is ', best_order
    plt.plot(orders, average_order)
    plt.title('BIC scores vs order')
    plt.xlabel('Number of components in GMM')
    plt.ylabel('BIC score')
    plt.show()
    return best_order


def prepare_data(data_set):
    """ Given a data_set (list) split it into train and test, display stats.

    Arguments:
        data_set: list

    Returns:
        train_set, test_set
        """

    train_set, test_set = train_test_split(data_set)
    train_set = list(train_set)
    test_set = list(test_set)
    N = len(c_image_collection)
    print 'Images read from ', data_dir, ' Total of ', N, ' images read'
    print 'We have ', len(train_set), ' training samples and ', len(test_set), \
        'test samples'
    print ' In training set'
    training_labels = [i.label for i in train_set]
    labels = set(training_labels)
    for i in labels:
        print 'There are ', training_labels.count(i), ' training images in', \
            ' class ', i, ' class name ', classes[i]
    testing_labels = [i.label for i in test_set]
    labels = set(testing_labels)
    for i in labels:
        print 'There are ', testing_labels.count(i), ' testing images in', \
            ' class ', i, ' class name ', classes[i]
    return train_set, test_set

# ############# Main ###########################################
if __name__ == '__main__':
    print '-------------------------------------------------------------------'
    print 'Started Image classification using GMM'
    print 'unique run string is ', UNIQUE_RUN_NAME, ' Number of components is ', \
        GMM_COMPONENTS
    data_dir = '../../../data/two_class_full_size/'
    # data_dir = '../../../data/caltech_small/'
    # classes = [i for i in os.listdir(data_dir) if os.path.isdir(data_dir + i)]
    classes = ['color-field-painting', 'realism']
    print 'Getting data from ', os.path.abspath(data_dir)
    print 'The classes are ', classes
    c_image_collection = []  # This will hold all the image models
    for genre_i in classes:
        image_collection = io.ImageCollection(data_dir + genre_i + '/*.jpg')
    #    image_collection = image_collection[:MAX_IMAGE_PER_CLASS]  # limiter
        for image_i in image_collection:
            curr_GmmImageSIFT = GmmImageSIFT(image_i)
            # set the label for current image
            curr_GmmImageSIFT.label = classes.index(genre_i)
            c_image_collection.append(curr_GmmImageSIFT)

    # ############ Data preparation ################################
    train_set, test_set = prepare_data(c_image_collection)
    # GMM_COMPONENTS = model_order_selection(train_set)

    # Training
    train_size = len(train_set)
    i = 0
    print 'Training started'
    # Displaying progress bar
    bar = progressbar.ProgressBar(maxval=train_size, \
        widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    for c_image_i in train_set:
        c_image_i.fit()
        bar.update(i+1)
        i += 1
    bar.finish()

    print 'Testing started'
    # Testing
    test_size = len(test_set)
    i = 0
    bar = progressbar.ProgressBar(maxval=test_size, \
        widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    for test_image_i in test_set:
        # check the log probability of each trainining image with current data
        test_scores = [train_image_i.score(test_image_i)
                    for train_image_i in train_set]
        max_index = test_scores.index(max(test_scores))
        # assign label with the best fit to each of the model,
        # this is nearest neighbour method
        test_image_i.predicted_label = train_set[max_index].label
        bar.update(i+1)
        i += 1
    bar.finish()

    true_labels = [i.label for i in test_set]
    predicted_labels = [i.predicted_label for i in test_set]
    print 'The Classification report is '
    print classification_report(true_labels, predicted_labels)
    # Save the trained GMM
    all_variables = dir()
    cPickle.dump(all_variables, open("All_variables_" + UNIQUE_RUN_NAME + ".p", "wb"))
