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

############ Constants ################
# TODO Change this while testing increase
GMM_COMPONENTS = 50 # Number of GMM components in Mixture modeling.
GMM_CONVERGENCE_THRESHOLD = 0.1 # GMM convergence, when to stop fitting
MAX_IMAGE_PER_CLASS = 100 # Maximum images to read per class, limits
# too big data set from being read

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


############## Main ##########################

data_dir = 'C:/Users/akr156/Pictures/two_class_full_size/'
genres = ['realism', 'color-field-painting']
c_image_collection = [] # This will hold all the image models
for genre_i in genres:
    image_collection = io.ImageCollection(data_dir + genre_i + '/*.jpg')
    image_collection = image_collection[:MAX_IMAGE_PER_CLASS] # limiter
    for image_i in image_collection:
        curr_gmm_image = gmm_image(image_i)
        # set the label for current image
        curr_gmm_image.label = genres.index(genre_i)
        c_image_collection.append(curr_gmm_image)

######### Model order selection ############################
print 'Finding best model order by bic'
X = c_image_collection[0] # Pick the first one,
#better would be to average across
orders = [5,10,20,50,100]
bic_scores = [(X.fit(order=i)).gmm_model.bic(X) for i in orders]
print 'The bic scores are ', bic_scores





train_set, test_set = train_test_split(c_image_collection)
train_set = list(train_set)
test_set = list(test_set)
N = len(c_image_collection)
print 'Images read from ', data_dir, ' Total of ', N, ' images read'
print 'We have ', len(train_set), ' training samples and ', len(test_set), \
    'test samples'

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

