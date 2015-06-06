""" This is the place where young remembering machine learning classifiers are trained in various arts.
"""
__author__ = 'Abhishek Rao'

# Headers
import numpy as np
import idx2numpy
from sklearn.cross_validation import train_test_split
import os
import glob
from random import shuffle
import urllib
import tarfile
import RememberingMachine


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
    print '===================== Results ======================='
    print 'The mean F1 score among all the classes is ', np.mean(score_sheet)
    return np.mean(score_sheet)

def mnist_school(classifier, samples_limit = 5123):
    # Raw training, no caffe use.
    print 'MNIST training started.'
    mnist_file ='/home/student/Downloads/MNIST/train-images.idx3-ubyte'
    if os.path.isfile(mnist_file):
        train_arr = idx2numpy.convert_from_file(mnist_file)
    else:
        print 'Error, no file'
        return
    print 'Train array loaded, size is ',train_arr.shape
    label_file = '/home/student/Downloads/MNIST/train-labels.idx1-ubyte'
    label_arr = idx2numpy.convert_from_file(label_file)
    print 'Train labels loaded, size is ', label_arr.shape
    digits = set(label_arr)
    # Train for each digit
    for digit_i in digits:
        # binarize
        y = 1*(label_arr==digit_i)[:samples_limit]
        x_train = np.vstack([i.flatten() for i in train_arr[:samples_limit]])
        classifier.fit_from_caffe_features(x_train, y, 'MNIST_' + str(digit_i))
    print 'MNIST training done.'
    # Testing on last 1000 samples
    print 'MNIST testing started.'
    scores = []
    for digit_i in digits:
        # binarize
        y = 1*(label_arr==digit_i)[-1000:]
        x_train = np.vstack([i.flatten() for i in train_arr[-1000:]])
        scores.append(classifier.score(x_train, y))
    print 'The mean score for MNIST Task is ', np.mean(scores)


# ################### Imagenet #########################################
def download_imagenet_wnid(wnid, root_folder='./imagenet/'):
    """ Given a wnid download  it from Imagenet.
    :param wnid: string, imagenet wnid
    :return: path of folder created.
    """
    if not os.path.exists(root_folder):
        os.makedirs(root_folder)
    # check if folder already exists
    if not os.path.exists(root_folder + wnid):
        username = 'abhishekraok'
        with open('accesskey.txt','r') as afile:
            accesskey = afile.read().strip()
        url = 'http://www.image-net.org/download/synset?wnid=' + wnid + \
              '&username=' + username + '&accesskey=' + accesskey + '&release=latest&src=stanford'
        print 'Url to download is ', url
        # Check if file is already downloaded
        archive_file = root_folder + wnid+'.tar'
        if not os.path.exists(archive_file):
            urllib.urlretrieve(url, archive_file )
        else:
            print 'Archive file already exists.'
        print 'extracting..'
        try:
            tar = tarfile.open(archive_file)
        except tarfile.ReadError:
            print 'Error, could not open ', archive_file, ', skipping.'
            return None
        os.makedirs(root_folder + wnid)
        tar.extractall(path=root_folder+wnid+'/')
        tar.close()
        os.remove(archive_file)
        print 'Folder created wnid=', wnid
    else:
        print 'Folder already exists'
    return os.path.abspath(root_folder + wnid)


def get_wornet_dict():
    """
    Returns a dictionary from wordnet file, which should be
    located at ./imagenet/words.txt
    :return: dictionry, key = name, value = wnid
    """
    with open('./imagenet/words.txt') as wordfile:
        imagenet_word_list = wordfile.read()
    return dict([i.split('\t')[::-1] for i in imagenet_word_list.split('\n')])

def imagenet_school_KG(classifier, imagenet_words_list=None):
    """
    Kindgergarten of imagenet, where simple shapes are taught.

    Given a list of string corresponding to wordnet words, download and
    caffinate them.

    :param classifier: the brave classifier willing to learn.
    :param imagenet_words_list:  list of strings of imagenet words.
    :return:
    """
    print 'Welcoem to Imagenet KG school'
    if not imagenet_words_list:
        imagenet_words_list = ['circle, round', 'line', 'triangle', 'square',
                               'parallel', 'parallelogram' ]
    imagenet_dict = get_wornet_dict()
    wnid_list = [imagenet_dict[i] for i in imagenet_words_list]
    valid_folders = []
    root_folder = './imagenet/'
    for wnid in wnid_list:
        print 'Getting wnid', wnid
        folder_i = download_imagenet_wnid(wnid, root_folder)
        if folder_i:
            valid_folders.append(folder_i)
    # Caffinate it's parent directory
    # RememberingMachine.caffinate_directory(os.path.abspath(os.path.join(valid_folders[0], os.path.pardir)))
    folder_learner(classifier, root_folder, task_name_prefix='Imagenet_KG_')


# ################## END imagenet ##################################################
def folder_learner(classifier, root_folder, task_name_prefix, negatives_samples_ratio=2,
                   max_categories=None):
    """
    :param classifier: A classifer that has fit function.
    :param root_folder: directory which contains many classes
    :param task_name_prefix: Name of the task to add to all, string
    :param negatives_samples_ratio: amount of negative samples to use.
    :param max_categories: maximum amount of categories to load.
    :return:
    """
    print 'Folder training started with folder ', root_folder
    categories = [i for i in os.listdir(root_folder)
                  if os.path.isdir(os.path.join(root_folder, i))]
    print 'Categories are ', categories
    small_categories = categories[:max_categories] if max_categories is not None else categories
    # Hold one out teaching. For each category, that category is positive, rest are negative.
    # The negatives also consist of a background images folder.
    for category_i in small_categories:
        positive_list = glob.glob(root_folder + '/' + category_i + '/*.jpg')
        positive_list.extend(glob.glob(root_folder + '/' + category_i + '/*.JPEG'))
        negatives_list = []
        for other_category_i in categories:
            if other_category_i != category_i:
                negatives_list += glob.glob(root_folder + '/' + other_category_i + '/*.jpg')
                negatives_list += glob.glob(root_folder + '/' + other_category_i + '/*.JPEG')
        shuffle(negatives_list)
        positive_samples_count = len(positive_list)
        small_negative_list = negatives_list[:positive_samples_count * negatives_samples_ratio]
        small_negative_list.extend(glob.glob(
            '/home/student/Downloads/101_ObjectCategories/BACKGROUND_Google' + '/*.jpg'))
        x_total = positive_list + small_negative_list
        y = [1]*len(positive_list) + [0]*len(small_negative_list)
        x_train, x_test, y_train, y_test = train_test_split(x_total, y)
        task_name = task_name_prefix + category_i
        classifier.fit(x_train, y_train, task_name)

if __name__ == '__main__':
    pass
    # download_imagenet_wnid('n03032811')


