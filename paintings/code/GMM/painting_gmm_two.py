import numpy as np
from sklearn import mixture
import skimage.io as io
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn import cross_validation
from sklearn.metrics import accuracy_score, classification_report
import cPickle
import os
import uuid
from sklearn.metrics import confusion_matrix

############ Constants ################
# TODO Change this while testing increase
GMM_COMPONENTS = 10 # Number of GMM components in Mixture modeling.
GMM_CONVERGENCE_THRESHOLD = 0.1 # GMM convergence, when to stop fitting
MAX_IMAGE_PER_CLASS = 100 # Maximum images to read per class, limits
# too big data set from being read
UNIQUE_RUN_NAME = str(uuid.uuid4())
############ Classes ###################
class gmm_image():
    """Class where objects have image, label and GMM"""

    def __init__(self, image=None):
        """can initialize with an image"""
        if image != None:
            self.image = image
        else:
            self.image = image
        self.label = None # which class they belong
        self.predicted_label = None # for test image only, predict label

    def fit(self, order=None):
        """create and fit GMM to self"""
        reshaped_image = np.vstack([self.image[:,:,i].ravel()
                                    for i in range(3)]).T
        if order == None:
            self.gmm_model = mixture.GMM(n_components=GMM_COMPONENTS,
                                        thresh=GMM_CONVERGENCE_THRESHOLD)
        else:
            self.gmm_model = mixture.GMM(n_components=order,
                                        thresh=GMM_CONVERGENCE_THRESHOLD)
        self.gmm_model.fit(reshaped_image)
        return self

    def score(self, X):
        """Just reshaping and finding gmm score

        :X: object of class gmm_image"""
        reshaped_image = np.vstack([X.image[:,:,i].ravel()
                                    for i in range(3)]).T
        return sum(self.gmm_model.score(reshaped_image))

############ functions ######################
def visualize3d(input3darray,color='b'):
    """Scatter plot 3d array"""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(input3darray[:,0],input3darray[:,1],input3darray[:,2],c=color)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.draw()
    time.sleep(0.1)

def full_image_vis(image):
    """Given an image of (x,y,rgb), displays it, shows pixel scatter plot
    and then the gmm fit of it."""
    fig = plt.figure()
    plt.subplot(2,2,1)
    plt.imshow(image)
    plt.title('The image')
    reshaped_image = np.vstack([image[:,:,i].ravel() for i in range(3)]).T
    plt.subplot(2,2,2)
    ax = fig.add_subplot(222, projection='3d')
    ax.scatter(reshaped_image[::20,0],reshaped_image[::20,1],reshaped_image[::20,2])
    plt.title('The RGB values scatter plot')
    g = mixture.GMM(n_components=50)
    print 'fitting ...'
    g.fit(reshaped_image)
    print 'fitting done'
    ax = fig.add_subplot(223, projection='3d')
    ax.scatter(g.means_[:,0],g.means_[:,1],g.means_[:,2],c='r')
    plt.title('The GMM means of the RGB scatter plot')
    plt.show()

def model_order_selction(X):
    """
    Given an gmm_image X, reshape it and fit a GMM for it, check it's bic. Pick
    the highest. Return integer.

    Arguments:
        X: object of class gmm_image

    Return:
        Integer, highest bic score order
    """
    print 'Finding best model order by bic'
    orders = [2,3,4,5,8,10,20,50]
    reshaped_X = np.vstack([X.image[:,:,i].ravel()
                                for i in range(3)]).T
    bic_scores = [(X.fit(order=i)).gmm_model.bic(reshaped_X) for i in orders]
    best_order = orders[bic_scores.index(max(bic_scores))]
    print 'The bic scores are ', bic_scores, ' best is ', best_order
    plt.plot(orders,bic_scores)
    plt.title('BIC scores vs order')
    plt.xlabel('Number of components in GMM')
    plt.ylabel('BIC score')
    plt.show()
    return best_order

def prepare_data(data_set):
    """ Given a dataset (list) split it into train and test, display stats.

    Arguements:
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

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues,
                          target_names=['class_1','class_2']):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(iris.target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

############## Main ###########################################
print ' Started Image classification using GMM'
print ' unique run string is ', UNIQUE_RUN_NAME
data_dir = '../../data/two_class_full_size/'
#data_dir = '../../data/caltech_small/'
#classes = [ i for i in os.listdir(data_dir) if os.path.isdir(data_dir+i)]
classes = ['color-field-painting','realism']
print 'Getting data from ', os.path.abspath(data_dir)
print ' The classes are ', classes
c_image_collection = [] # This will hold all the image models
for genre_i in classes:
    image_collection = io.ImageCollection(data_dir + genre_i + '/*.jpg')
    image_collection = image_collection[:MAX_IMAGE_PER_CLASS] # limiter
    for image_i in image_collection:
        curr_gmm_image = gmm_image(image_i)
        # set the label for current image
        curr_gmm_image.label = classes.index(genre_i)
        c_image_collection.append(curr_gmm_image)

#X = c_image_collection[6] # Pick the first one for finding model order selction.
#better would be to average across
#GMM_COMPONENTS = model_order_selction(X)

############# Data preparation ################################

train_set, test_set = prepare_data(c_image_collection)

# Training
print 'Training started'
for c_image_i in train_set:
    c_image_i.fit()

print 'Testing started'
# Testing
for test_image_i in test_set:
    # check the log probability of each trainining image with current data
    test_scores = [train_image_i.score(test_image_i)
                   for train_image_i in train_set]
    max_index = test_scores.index(max(test_scores))
    # assign label with the best fit to each of the model,
    # this is nearest neighbour method
    test_image_i.predicted_label = train_set[max_index].label

true_labels = [i.label for i in test_set]
predicted_labels = [i.predicted_label for i in test_set]
print 'The Classification report is '
print classification_report(true_labels, predicted_labels)

# Save the trained GMM
#pickle.dump(train_set, open("train_set"+UNIQUE_RUN_NAME+ ".p","wb"))
