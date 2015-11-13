""" This is the place where young remembering machine learning classifiers are trained in various arts.
"""
__author__ = 'Abhishek Rao'

# Headers
import glob
import os
import tarfile
import urllib
from random import shuffle

import idx2numpy
import numpy as np
from sklearn.cross_validation import train_test_split
# from RememberingMachine import meanie, dot_with_11
import string
import random


def convert_to_valid_pathname(filename):
    validfilenamechars = "-_.()%s%s" % (string.ascii_letters, string.digits)
    # cleanedfilename = unicodedata.normalize('NFKD', filename).encode('ASCII', 'ignore')
    return ''.join(c for c in filename if c in validfilenamechars)


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
        y = [1] * len(positive_list) + [0] * len(small_negative_list)
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
        small_negative_list = negatives_list[:positive_samples_count * 2]
        x_total = positive_list + small_negative_list
        y = [1] * len(positive_list) + [0] * len(small_negative_list)
        x_train, x_test, y_train, y_test = train_test_split(x_total, y)
        score = classifier.score(x_test, y_test)
        print 'In the category ', category_i, ' F1 score is ', score
        score_sheet.append(score)
    print '===================== Results ======================='
    print 'The mean F1 score among all the classes is ', np.mean(score_sheet)
    return np.mean(score_sheet)


def mnist_school(classifier, samples_limit=5123):
    # Raw training, no caffe use.
    print 'MNIST training started.'
    mnist_file = '/home/student/Downloads/MNIST/train-images.idx3-ubyte'
    if os.path.isfile(mnist_file):
        train_arr = idx2numpy.convert_from_file(mnist_file)
    else:
        print 'Error, no file'
        return
    print 'Train array loaded, size is ', train_arr.shape
    label_file = '/home/student/Downloads/MNIST/train-labels.idx1-ubyte'
    label_arr = idx2numpy.convert_from_file(label_file)
    print 'Train labels loaded, size is ', label_arr.shape
    digits = set(label_arr)
    # Train for each digit
    for digit_i in digits:
        # binarize
        y = 1 * (label_arr == digit_i)[:samples_limit]
        x_train = np.vstack([i.flatten() for i in train_arr[:samples_limit]])
        classifier.fit_from_features(x_train, y, 'MNIST_' + str(digit_i))
    print 'MNIST training done.'
    # Testing on last 1000 samples
    print 'MNIST testing started.'
    scores = []
    for digit_i in digits:
        # binarize
        y = 1 * (label_arr == digit_i)[-1000:]
        x_train = np.vstack([i.flatten() for i in train_arr[-1000:]])
        scores.append(classifier.score(x_train, y))
    print 'The mean score for MNIST Task is ', np.mean(scores)


# ################### Imagenet #########################################
def download_imagenet_wnid(wnid, actual_label, root_folder='./imagenet/'):
    """ Given a wnid download  it from Imagenet.
    :param wnid: string, imagenet wnid
    :return: path of folder created.
    """
    if not os.path.exists(root_folder):
        os.makedirs(root_folder)
    # check if folder already exists
    created_folder = root_folder + wnid + '_' + convert_to_valid_pathname(actual_label)
    if not os.path.exists(created_folder):
        username = 'abhishekraok'
        with open('accesskey.txt', 'r') as afile:
            accesskey = afile.read().strip()
        url = 'http://www.image-net.org/download/synset?wnid=' + wnid + \
              '&username=' + username + '&accesskey=' + accesskey + '&release=latest&src=stanford'
        print 'Url to download is ', url
        # Check if file is already downloaded
        archive_file = root_folder + wnid + '.tar'
        if not os.path.exists(archive_file):
            urllib.urlretrieve(url, archive_file)
        else:
            print 'Archive file already exists.'
        print 'extracting..'
        try:
            tar = tarfile.open(archive_file)
        except tarfile.ReadError:
            print 'Error, could not open ', archive_file, ', skipping.'
            os.remove(archive_file)
            return None
        os.makedirs(created_folder)
        tar.extractall(path=created_folder + '/')
        tar.close()
        os.remove(archive_file)
        print 'Folder created name ', created_folder
    else:
        print 'Folder already exists'
    return created_folder


def get_wornet_dict():
    """
    Returns a dictionary from wordnet file, which should be
    located at ./imagenet/words.txt
    :return: dictionry, key = name, value = wnid
    """
    with open('./imagenet/words.txt') as wordfile:
        imagenet_word_list = wordfile.read()
    return dict([i.split('\t')[::-1] for i in imagenet_word_list.split('\n')])


# def imagenet_class_KG(classifier, imagenet_words_list=None):
#     """
#     Kindgergarten of imagenet, where simple shapes are taught.
#
#     Given a list of string corresponding to wordnet words, download and
#     caffinate them.
#
#     :param classifier: the brave classifier willing to learn.
#     :param imagenet_words_list:  list of strings of imagenet words.
#     :return:
#     """
#     print 'Welcoem to Imagenet KG class'
#     if not imagenet_words_list:
#         imagenet_words_list = ['circle, round', 'line', 'parallel']
#     imagenet_dict = get_wornet_dict()
#     wnid_list = [imagenet_dict[i] for i in imagenet_words_list]
#     valid_folders = []
#     root_folder = './imagenet/'
#     # list_remove = [(j,convert_to_valid_pathname(i)) for j,i in zip(wnid_list,imagenet_words_list)]
#     for actual_label, wnid in zip(imagenet_words_list, wnid_list):
#         print 'Getting wnid', wnid
#         folder_i = download_imagenet_wnid(wnid, actual_label, root_folder)
#         if folder_i:
#             valid_folders.append(folder_i)
#     # Caffinate it's parent directory
#     # RememberingMachine.caffinate_directory(os.path.abspath(os.path.join(valid_folders[0], os.path.pardir)))
#     folder_learner(classifier, root_folder, task_name_prefix='Imagenet0_')
#

def random_imagenet_learner(classifier, k_words=None, number_labels=10, feedback_remember=True, cleanup=True):
    """
    Calls imagenet class again and again with progressively tougher objects.
    :param classifier: the willing learning classifier
    :param k_words: the list of imagenet labels to learn.
    :param cleanup: default True, will delete the downloaded folder.
    :return:
    """
    print 'Welcoem to Imagenet random class'
    imagenet_dict = get_wornet_dict()
    words_list = imagenet_dict.keys()
    root_folder = './imagenet/'
    if not k_words:
        k_words = random.sample(words_list, number_labels)
    wnid_list = []
    new_k_words = []
    for i_k in k_words:
        try:
            wnid_list.append(imagenet_dict[i_k])
            new_k_words.append(i_k)
        except KeyError:
            print 'The word ', i_k, ' does not exist in Imagenet.'
    # Download
    downloaded_folders_list = []
    for actual_label, wnid in zip(new_k_words, wnid_list):
        print 'Getting wnid', wnid, actual_label
        downloaded_folders_list.append(download_imagenet_wnid(wnid, actual_label, root_folder))
    # learn actually
    for word_i in downloaded_folders_list:
        if word_i:
            single_label_learner(classifier, root_folder=root_folder, label=os.path.split(word_i)[-1],
                                 task_name_prefix='imagenet_rn_', remembering_threshold=0.8, max_samples_per_cat=1000,
                                 feedback_remember=feedback_remember, download=False)
    # delete folders to save space
    if cleanup:
        import shutil
        for word_i in downloaded_folders_list:
            if word_i:
                shutil.rmtree(word_i)
                print 'Deleted folder ', word_i



# ################## END imagenet ##################################################
def folder_learner(classifier, root_folder, task_name_prefix, negatives_samples_ratio=2,
                   max_categories=None, use_background=False, max_samples_per_cat=None,
                   remembering_threshold=1.1, feedback_remember=False):
    """
    :param classifier: A classifer that has fit function.
    :param root_folder: directory which contains many classes
    :param task_name_prefix: Name of the task to add to all, string
    :param negatives_samples_ratio: amount of negative samples to use.
    :param max_categories: maximum amount of categories to load.
    :param use_background: boolean, default True, if False, won't use the google background
        images to extend negatives list.
    :param max_samples_per_cat: maximum number of samples to load in positive class. Default
        None, which means all.
    :param remembering_threshold: applies when feedback_remember is True.
        the F1 score above which to save the classifier. By default
        this value is 1.1 meaning it wont save no matter what. A value less that zero will always
        save. 0.8 would save only if it's sure.
    :param feedback_remember: boolean, default False. If true saves or reload based on score.
    :return:
    """
    print 'Folder training started with folder ', root_folder
    categories = [i for i in os.listdir(root_folder)
                  if os.path.isdir(os.path.join(root_folder, i))]
    print 'Categories are ', categories
    small_categories = categories[:max_categories] if max_categories is not None else categories
    # Hold one out teaching. For each category, that category is positive, rest are negative.
    # The negatives also consist of a background images folder.
    all_score = []
    for category_i in small_categories:
        positive_list = glob.glob(root_folder + '/' + category_i + '/*.jpg')
        positive_list.extend(glob.glob(root_folder + '/' + category_i + '/*.JPEG'))
        positive_list.extend(glob.glob(root_folder + '/' + category_i + '/*.png'))
        # limit number of samples.
        if max_samples_per_cat:
            positive_list = positive_list[:max_samples_per_cat]
            predicted_1s = np.zeros((len(small_categories), max_samples_per_cat))
        positive_samples_count = len(positive_list)
        negatives_list = []
        for other_category_i in categories:
            if other_category_i != category_i:
                negatives_list += glob.glob(root_folder + '/' + other_category_i + '/*.jpg')
                negatives_list += glob.glob(root_folder + '/' + other_category_i + '/*.JPEG')
                negatives_list += glob.glob(root_folder + '/' + other_category_i + '/*.png')
                # no then it will only load the first few negatives always.
                # if len(negatives_list) > positive_samples_count * negatives_samples_ratio:
                #     break  # enough negatives collected.
        shuffle(negatives_list)
        small_negative_list = negatives_list[:positive_samples_count * negatives_samples_ratio]
        if use_background:
            small_negative_list.extend(glob.glob(
                '/home/student/Downloads/101_ObjectCategories/BACKGROUND_Google' + '/*.jpg'))
        x_total = positive_list + small_negative_list
        y = [1] * len(positive_list) + [0] * len(small_negative_list)
        x_train, x_test, y_train, y_test = train_test_split(x_total, y)
        task_name = task_name_prefix + category_i
        print 'Currently training category ', category_i, ' with number of samples = ', len(x_total)
        classifier.fit(x_train, y_train, task_name)
        score = classifier.score(x_test, y_test)
        print 'The test score for this task is ', score
        all_score.append(score)
        if feedback_remember:
            if score > remembering_threshold:
                print 'This task was learn well. Classifier shall remember.'
                classifier.save()
                # else:
                #     # forget it, can't differentiate well
                #     classifier.remove(task_name)
            else:
                print ' :( classifier doesn\'t understand ', category_i, ' task at all. Forget it. Reload'
                classifier.reload()
    print 'The mean F1 score (unweighted) is ', np.mean(all_score)


def single_label_learner(classifier, root_folder, label, task_name_prefix, negatives_samples_ratio=2,
                   use_background=False, max_samples_per_cat=None,
                   remembering_threshold=1.1, feedback_remember=False, download=True):
    """
    :param classifier: A classifer that has fit function.
    :param root_folder: directory which contains many classes
    :param task_name_prefix: Name of the task to add to all, string
    :param negatives_samples_ratio: amount of negative samples to use.
    :param use_background: boolean, default True, if False, won't use the google background
        images to extend negatives list.
    :param max_samples_per_cat: maximum number of samples to load in positive class. Default
        None, which means all.
    :param remembering_threshold: applies when feedback_remember is True.
        the F1 score above which to save the classifier. By default
        this value is 1.1 meaning it wont save no matter what. A value less that zero will always
        save. 0.8 would save only if it's sure.
    :param feedback_remember: boolean, default False. If true saves or reload based on score.
    :param download: boolean, default True. Whether to download image from search engine.
    :return:
    """
    print 'Single folder training started for the label', label
    categories = [i for i in os.listdir(root_folder)
                  if os.path.isdir(os.path.join(root_folder, i))]
    if download:
        get_images_bing(query=label)
    # Hold one out teaching. For each category, that category is positive, rest are negative.
    # The negatives also consist of a background images folder.
    all_score = []
    # remove the ending /
    if root_folder[-1] == '/':
        root_folder = root_folder[:-1]
    positive_list = glob.glob(root_folder + '/' + label + '/*.jpg')
    positive_list.extend(glob.glob(root_folder + '/' + label + '/*.JPEG'))
    positive_list.extend(glob.glob(root_folder + '/' + label + '/*.png'))
    # limit number of samples.
    if max_samples_per_cat:
        positive_list = positive_list[:max_samples_per_cat]
    positive_samples_count = len(positive_list)
    if positive_samples_count < 50:
        print 'There are less than 50 samples of image in this, wont learn ', label
        return
    negatives_list = []
    for other_category_i in categories:
        if other_category_i != label:
            negatives_list += glob.glob(root_folder + '/' + other_category_i + '/*.jpg')
            negatives_list += glob.glob(root_folder + '/' + other_category_i + '/*.JPEG')
            negatives_list += glob.glob(root_folder + '/' + other_category_i + '/*.png')
            # no then it will only load the first few negatives always.
            # if len(negatives_list) > positive_samples_count * negatives_samples_ratio:
            #     break  # enough negatives collected.
    shuffle(negatives_list)
    small_negative_list = negatives_list[:positive_samples_count * negatives_samples_ratio]
    if use_background:
        small_negative_list.extend(glob.glob(
            '/home/student/Downloads/101_ObjectCategories/BACKGROUND_Google' + '/*.jpg'))
    print 'There are ', len(positive_list), ' positive samples and ', len(small_negative_list), ' negatives.'
    x_total = positive_list + small_negative_list
    y = [1] * len(positive_list) + [0] * len(small_negative_list)
    x_train, x_test, y_train, y_test = train_test_split(x_total, y)
    task_name = task_name_prefix +  label
    print 'Currently training category ', label, ' with number of samples = ', len(x_total)
    classifier.fit(x_train, y_train, task_name)
    score = classifier.score(x_test, y_test)
    print 'The test score for this task is ', score
    all_score.append(score)
    if feedback_remember:
        if score > remembering_threshold:
            print 'This task was learn well. Classifier shall remember.'
            classifier.save()
            # else:
            #     # forget it, can't differentiate well
            #     classifier.remove(task_name)
        else:
            print ' :( classifier doesn\'t understand ', label, ' task at all. Forget it. Reload'
            classifier.reload()
    return score


# Linear trainer
def train_square(classifier):
    # Task 1
    # create a sample dataset. centred at 0,4 and 4,4. Normally distributed
    # and variance 1. 100 samples and 2 dimension.
    x_0 = np.random.randn(100, 2) + np.array([-4, 0])
    x_1 = np.random.randn(100, 2) + np.array([4, 0])
    y = np.array([0] * 100 + [1] * 100)
    x_total = np.vstack([x_0, x_1])
    classifier.fit(x_total, y, 'Task1: of -4,0 and 4,0')
    classifier.visualize_clf(x_total, y)
    # Task 2
    x_0 = np.random.randn(100, 2) + np.array([0, 0])
    x_1 = np.random.randn(100, 2) + np.array([12, 0])
    y = np.array([0] * 100 + [1] * 100)
    x_total = np.vstack([x_0, x_1])
    classifier.fit(x_total, y, 'Task2: of 0,0 and 12,0')
    classifier.visualize_clf(x_total, y)
    # Task 3
    # create a sample dataset. centred at 0,0 and 0,4. Normally distributed
    # and variance 1. 100 samples and 2 dimension.
    x_0 = np.random.randn(100, 2) + np.array([0, -4])
    x_1 = np.random.randn(100, 2) + np.array([0, 4])
    y = np.array([0] * 100 + [1] * 100)
    x_total = np.vstack([x_0, x_1])
    classifier.fit(x_total, y, 'Task3: of 0,-4 and 0,4')
    classifier.visualize_clf(x_total, y)
    # Task 4
    x_0 = np.random.randn(100, 2) + np.array([0, 0])
    x_1 = np.random.randn(100, 2) + np.array([0, 12])
    y = np.array([0] * 100 + [1] * 100)
    x_total = np.vstack([x_0, x_1])
    classifier.fit(x_total, y, 'Task4: of 0,0 and 0,12')
    classifier.visualize_clf(x_total, y)
    # Task 5, the square task.
    x_0 = np.random.randn(100, 2) + np.array([4, 4])
    x_1 = np.random.randn(25, 2) + np.array([-2, 4])
    x_2 = np.random.randn(25, 2) + np.array([4, 10])
    x_3 = np.random.randn(25, 2) + np.array([4, -2])
    x_4 = np.random.randn(25, 2) + np.array([10, 4])
    x_5 = np.vstack([x_1, x_2, x_3, x_4])
    y = np.array([0] * 100 + [1] * 100)
    x_total = np.vstack([x_0, x_5])
    classifier.fit(x_total, y, 'Task5: of far points and 4,4')
    classifier.visualize_clf(x_total, y)


def train_tri_band(classifier):
    # Task 1
    # create a sample dataset. centred at 0,4 and 4,4. Normally distributed
    # and variance 1. 100 samples and 2 dimension.
    x_0 = np.random.randn(100, 2) + np.array([-4, 0])
    x_1 = np.random.randn(100, 2) + np.array([4, 0])
    y = np.array([0] * 100 + [1] * 100)
    x_total = np.vstack([x_0, x_1])
    classifier.fit(x_total, y, 'Task1: of -4,0 and 4,0')
    # classifier.visualize_clf(x_total, y)

    # Task 2
    x_0 = np.random.randn(100, 2) + np.array([4, 0])
    x_1 = np.random.randn(100, 2) + np.array([14, 0])
    y = np.array([1] * 100 + [0] * 100)
    x_total = np.vstack([x_0, x_1])
    classifier.fit(x_total, y, 'Task2: of 2,0 and 14,0')
    # classifier.visualize_clf(x_total, y)

    # Task 3, the tri band task.
    x_0 = np.random.randn(100, 2) + np.array([4, 0])
    x_1 = np.random.randn(50, 2) + np.array([-4, 0])
    x_2 = np.random.randn(50, 2) + np.array([14, 0])
    x_5 = np.vstack([x_1, x_2])
    y = np.array([0] * 100 + [1] * 100)
    x_total = np.vstack([x_0, x_5])
    classifier.fit(x_total, y, 'Task5: of far points and 4,0')
    # classifier.visualize_clf(x_total, y)


def random_linear_trainer(classifier, stages=10, visualize=False):
    for stage_i in range(stages):
        center_1 = np.random.randn(1, 2) * 10
        center_2 = np.random.randn(1, 2) * 10
        x_0 = np.random.randn(100, 2) + center_1
        x_1 = np.random.randn(100, 2) + center_2
        y = np.array([0] * 100 + [1] * 100)
        x_total = np.vstack([x_0, x_1])
        task_name = 'Random Linear ' + str(center_1) + ' and ' + str(center_2)
        classifier.fit(x_total, y, task_name)
        if visualize and stage_i % 5 == 0:
            classifier.visualize_clf(x_total, y)


def random_linear_trainer2(classifier, stages=10, visualize=False):
    for stage_i in range(stages):
        center_1 = np.random.randn(1, 2) * 10
        center_2 = np.random.randn(1, 2) * 10
        center_3 = np.random.randn(1, 2) * 10
        x_0 = np.random.randn(100, 2) + center_1
        x_1 = np.random.randn(50, 2) + center_2
        x_2 = np.random.randn(50, 2) + center_3
        x_5 = np.vstack([x_1, x_2])
        y = np.array([0] * 100 + [1] * 100)
        x_total = np.vstack([x_0, x_5])
        task_name = 'Random Linear 2' + str(center_1) + \
                    ' vs ' + str(center_2) + str(center_3)
        classifier.fit(x_total, y, task_name)
        if stage_i % 5 == 0 and visualize:
            classifier.visualize_clf(x_total, y)


def growing_complex_trainer(classifier, repeat_per_cluster=10, number_of_clusters=6):
    # start with classifying two classes that have 3 clusters each.
    for clusters_count in range(3, number_of_clusters, 1):
        # repeat this many times at each cluster count.
        for repetition_i in range(repeat_per_cluster):
            centers_1 = np.random.randn(clusters_count + 1, 2) * 10
            centers_2 = np.random.randn(clusters_count + 1, 2) * 10
            x_0 = []  # points for class 0
            x_1 = []  # points for class 1
            for center_i in range(clusters_count + 1):
                x_0.append(np.random.randn(100, 2) + centers_1[center_i, :])
                x_1.append(np.random.randn(100, 2) + centers_2[center_i, :])
            x_0_np = np.vstack(x_0)
            x_1_np = np.vstack(x_1)
            y = np.array([0] * x_0_np.shape[0] + [1] * x_1_np.shape[0])
            x_total = np.vstack([x_0_np, x_1_np])
            task_name = 'Growing complex trainer ' + str(clusters_count) + \
                        ' rep ' + str(repetition_i)
            classifier.fit(x_total, y, task_name)
        classifier.visualize_clf(x_total, y)
        print 'Score is ', classifier.score(x_total, y)


# baby AI
def amat_to_numpy(amat_file):
    from PIL import Image

    folder_name = amat_file[:-11] + '_real'
    if not os.path.isdir(folder_name):
        os.makedirs(folder_name)
    if not os.path.isdir(folder_name + '/rectangle'):
        os.makedirs(folder_name + '/rectangle')
        os.makedirs(folder_name + '/ellipse')
        os.makedirs(folder_name + '/triangle')
    shapes = ['rectangle', 'ellipse', 'triangle']
    with open(amat_file, 'r') as fp:
        amat_data = fp.read()
        in_lines = amat_data.split('\n')
        count = 0
        for line_i in in_lines[1:-1]:
            chars = line_i.split(' ')
            fs = [256 * float(i) for i in chars[:1024]]
            im = Image.new('L', (32, 32))
            im = im.putdata(fs)
            shape_i = shapes[int(chars[1024])]
            file_name = folder_name + '/' + shape_i + '/' + \
                        str(count) + '.png'
            with open(file_name, 'w') as fp:
                im.save(fp, 'png')
            count += 1


def cifar_convert_folder(cifar_10_folder):
    """
    Once you download the cifar dataset from
    http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
    and extract it, this converts it into a folder full of images.

    :param cifar_10_folder: the folder where the above tar is extracted.
    :return:
    """
    print 'processing cifar 10 data'

    def unpickle_cifar(file):
        import cPickle

        fo = open(file, 'rb')
        dict = cPickle.load(fo)
        fo.close()
        return dict

    def dict2folder(in_dict, root_folder):
        npdata = in_dict['data']
        labels = in_dict['labels']
        file_names = in_dict['filenames']
        set_labels = set(labels)
        for label_i in set_labels:
            if not os.path.isdir(cifar_10_folder + str(label_i)):
                os.makedirs(cifar_10_folder + str(label_i))
        from PIL import Image

        for i_data, i_label, i_file in zip(npdata, labels, file_names):
            reshaped_image = i_data.reshape(3, 32, 32).T
            pilim = Image.fromarray(reshaped_image)
            with open(root_folder + str(i_label) + '/' + i_file, 'w') as fp:
                pilim.save(fp, 'png')

    file_list = os.listdir(cifar_10_folder)
    dicts_list = [unpickle_cifar(i) for i in file_list[2:6] + file_list[7:]]
    [dict2folder(i, cifar_10_folder) for i in dicts_list]
    print 'Done.'


def cifar_school(classifier):
    """
    Teach the classifier to identify cifar dataset.
    :param classifier:  The classifier
    :return:
    """
    cifar_10_folder = '/home/student/Downloads/cifar-10-batches-py/'
    folder_learner(classifier, cifar_10_folder, task_name_prefix='cifar10_')


def painting_school(classifier, max_samples_per_cat):
    """
    Teach the classifier to identify cifar dataset.
    :param classifier:  The classifier
    :param max_samples_per_cat: maximum number of samples to load in positive class. Default
        None, which means all.
    :return:
    """
    paintings_folder = '/home/student/Lpromising-patterns/paintings/data/two_class_full_size'
    folder_learner(classifier, paintings_folder, task_name_prefix='wikipainting_',
                   use_background=False, max_samples_per_cat=max_samples_per_cat)


def paint_experiment(classifier, data_dir=None, max_train_samples=None):
    """
    Perform multiclass classification on the folder.

    :param classifier: any scikit style classifier
    :param data_dir:  the path of data where each class images are in a separate folder.
    :return: prints classification report.
    """
    # load data
    if not data_dir:
        data_dir = '/home/student/ln_onedrive/code/promising-patterns/paintings/data/five_class_full_size/'
    classes = [i for i in os.listdir(data_dir) if os.path.isdir(data_dir + i)]
    print 'Getting data from ', os.path.abspath(data_dir)
    print 'The classes are ', classes
    folder_labels = []  # indicates which class it belongs to
    x_files = []
    for class_count, class_name in enumerate(classes):  # For each genre
        current_genre_file_list = glob.glob(data_dir + class_name + '/*.jpg')  # List all the filenames
        x_files.extend(current_genre_file_list)
        folder_labels.extend([class_count] * len(current_genre_file_list))
    x_train, x_test, y_train, y_test = train_test_split(x_files, folder_labels)
    if max_train_samples:
        x_train = x_train[:max_train_samples]
        y_train = y_train[:max_train_samples]
    print 'There are ', len(x_train), ' training samples. Starting classification...'

    # multiclass classification begin
    y_pred = [0] * len(y_test)
    from sklearn.metrics import classification_report
    for class_count, class_name in enumerate(classes):  # For each genre
        # convert from 1...N into binary for each class
        y_current_train = 1*[i == class_count for i in y_train]
        classifier.fit(x_train, y_current_train, 'paintings_' + class_name, save_classifier=False)
        y_current_pred = classifier.predict(x_test)
        y_current_test = 1*[i == class_count for i in y_test]
        # print(classification_report(y_current_test, y_current_pred, target_names=['rest', class_name]))
        # current from binary into 1..N
        for i in range(len(y_current_test)):
            if y_current_pred[i]:
                y_pred[i] = class_count
    print(classification_report(y_test, y_pred, target_names=classes))


def bing_learner(classifier, words_list=None, download_folder='bing_images', feedback_remember=True):
    """
    Given a list of words, download it from bing and learn to classify

    :param classifier:
    :param words_list:
    :return:
    """
    words_list = ['points', 'zigzag', 'spherical', 'ellipse', 'square', 'rectangle', 'colorful',
                  'smooth', 'rough', 'many', 'sparse', 'plain'] if words_list is None else words_list
    for word_i in words_list:
        get_images_bing(word_i, root_folder=download_folder)
    for word_i in words_list:
        single_label_learner(classifier, root_folder=download_folder, label=word_i, task_name_prefix='bingl_',
                             remembering_threshold=0.8, feedback_remember=feedback_remember)


def bing_long_learner(classifier):
    words_list = ['edge', 'sharp', 'smooth', 'texture', 'polka', 'curvy', 'apple', 'leaf',
                  'sun', 'moon', 'face', 'human', 'hands', 'oval']
    bing_learner(classifier, words_list=words_list, feedback_remember=True)

def get_images_bing(query, root_folder='bing_images'):
    print 'Getting images of ', query, ' from bing.'
    from bs4 import BeautifulSoup
    import requests
    import re
    import urllib2
    import os

    if not os.path.isdir(root_folder):
        os.makedirs(root_folder)
    url = "http://www.bing.com/images/search?q=" + query + \
          "&qft=+filterui:photo-photo+filterui:imagesize-small&FORM=R5IR3"
    # adding minor variations in the query to get more images
    url2 = "http://www.bing.com/images/search?q=images of " + query + \
          "&qft=+filterui:photo-photo+filterui:imagesize-small&FORM=R5IR3"
    url3 = "http://www.bing.com/images/search?q=" + query + \
          " photos&qft=+filterui:photo-photo+filterui:imagesize-small&FORM=R5IR3"
    url4 = "http://www.bing.com/images/search?q=photo of " + query + \
          "&qft=+filterui:photo-photo+filterui:imagesize-small&FORM=R5IR3"
    images = []
    for url_i in [url, url2, url3, url4]:
        soup = BeautifulSoup(requests.get(url).text)
        images += [a['src'] for a in soup.find_all("img", {"src": re.compile("mm.bing.net")})]
        images += [a['src2'] for a in soup.find_all("img", {"src2": re.compile("mm.bing.net")})]
    current_folder = root_folder + '/' + query
    # remove duplicates
    images = set(images)
    if not os.path.isdir(current_folder):
        os.makedirs(current_folder)
        for i, img in enumerate(images):
            raw_img = urllib2.urlopen(img).read()
            f = open(current_folder + "/" + str(i) + ".jpg", 'wb')
            f.write(raw_img)
            f.close()
        print 'Saved ', len(images), ' images '
    else:
        print 'The folder ', current_folder, ' already exist, not doing anything.'


def go_to_all_schools(classifier):
    """
    Send to all schools availalble

    :param classifier:
    :return:
    """
    print 'The schools available are caltech_101 , cifar10, paintings'
    bing_learner(classifier)
    caltech_101(classifier)
    cifar_school(classifier)
    painting_school(classifier)
    random_imagenet_learner(classifier)


if __name__ == '__main__':
    pass
    # download_imagenet_wnid('n03032811')
    # amat_to_numpy('/home/student/Downloads/shapeset/shapeset1_1cs_2p_3o.5000.valid.amat')
