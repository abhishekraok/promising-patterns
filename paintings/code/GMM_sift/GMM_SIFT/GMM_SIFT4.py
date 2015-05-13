# coding: utf-8
# # Gaussian Mixture Model Dense SIFT
# We will try to classify our data by extracting Dense SIFT feature from each image and buidling Gaussian Mixture Model (GMM) for them.
__author__ = 'Abhishek Rao'

# Loading all the headers
import numpy as np
from sklearn import mixture
from glob import glob
import os
print os.getcwd() # Just checking current directory
print 'File exists', os.path.isfile('../code/GMM_sift/GMM_SIFT/GMM_SIFT.py')
import sys
sys.path.append('../code/GMM_sift/GMM_SIFT/')
import cv2
from sklearn.cross_validation import train_test_split
from sklearn.metrics import average_precision_score, classification_report

# Constants
NUM_COMPONENT = 128  # number of components in GMM

# Functions
def extract_dense_SIFT_feat(filename):
    """ Given a filename extracts the dense SIFT features.
    Argument:
        filename: string path to file
    """
    im = cv2.imread(filename)
    dense=cv2.FeatureDetector_create("Dense")
    kp=dense.detect(im)
    sift=cv2.SIFT()
    kp,des=sift.compute(im,kp)
    return des

def train(train_files,train_labels):
    """Train on given dataset"""
    # Start training
    print 'Training started'
    number_of_classes = len(set(train_labels))
    train_features = []
    for image_i in train_files:
        SIFT_features = extract_dense_SIFT_feat(image_i)
        train_features.append(SIFT_features)
    X_train = [np.zeros([1,128])]*number_of_classes
    for i,label_i in enumerate(train_labels):
        X_train[label_i] = np.vstack([X_train[label_i],train_features[i]])
    # Create a GMM for each class
    Mixture_list = [] # Empty list of mixture
    for class_i in X_train:
        mixture_model = mixture.GMM(n_components=NUM_COMPONENT)
        print 'This class has training data of shape', class_i[1:,:].shape
        mixture_model.fit(class_i[1:,:]) # Ignore 1st row as it is zeros
        Mixture_list.append(mixture_model)
    print 'Training  done'
    return Mixture_list

def predict(mixtures_list, test_files):
    """Give the predicted output.
    Arguments:
    mixtures_list : List of gmm
    test_files: list of image file paths for testing.
    """
    print 'Prediction Started'
    y_pred = [] # The to be returned predicted results
    for test_i in test_files:
        test_feat = extract_dense_SIFT_feat(test_i)
        class_scores = [sum(i.score(test_feat)) for i in mixtures_list]
        y_pred.append(np.argmax(class_scores))
    return y_pred

def get_all_jpgs_list(data_dir,subdirectories,maxlen):
    """Given a director = data_dir,
    and a list of subdirectories name in that
    gives a list with filename of all .jpg files in that inclluding subdirectory.

    Arguments:
        data_dir: path where all the subdirectories are located
        subdirectories: list of names of subdirectories"""
    files_list = []
    labels_list = []
    for current_class in subdirectories:  # For each genre
        current_folder_files = glob(data_dir + current_class + '/*.jpg')
        if maxlen != None:
            current_folder_files = current_folder_files[:maxlen]
        files_list.extend(current_folder_files)  # List all the filenames
        print 'Currently in Folder No.', subdirectories.index(current_class)
        labels_list.extend([subdirectories.index(current_class)]*len(current_folder_files))
    return (files_list,labels_list)

def SIFT_experiment(data_dir, classes=['color-field-painting', 'expressionism'], maxlen=None):
    """Given classes perform SIFT feature image classification experiment"""
    # Get all files list
    jpg_files_list, labels_list = get_all_jpgs_list(data_dir,classes,maxlen)
    # Split into train and test set
    train_files, test_files, train_labels, test_labels = train_test_split(jpg_files_list, labels_list)
    # Train
    Mixtures_list = train(train_files,train_labels)
    # Test
    y_pred = predict(Mixtures_list,test_files)
    if len(set(test_labels)) == 2:
        print 'Average precision = ', average_precision_score(test_labels, y_pred)
    print(classification_report(test_labels, y_pred, target_names=classes))


data_dir = '../data/two_class_full_size/'
clf = SIFT_experiment(data_dir,maxlen=100)
