"""
Code to get paintings from web and convert it into numpy file.

Date: 26 Oct 2014
Author: Abhishek Rao

The general algorithm is as follows:

1. List all styles you want
2. For each style create a folder
3. Crawl that style and get list of links for images
4. For each link
5. Download the links using urllib and convert to PIL image, resize (optional)
   and write to JPG file.
6. For each image convert to numpy array and append the folder index to create
   a matrix where the rows are samples, columns = features + label,
   hence the number of columns = number of features + 1. Save this numpy matrix
   as csv file, create two, one for train, and one for test.

There is also a utility to visualize the downloaded data.
"""
__docformat__ = 'restructedtext en'

import numpy
from bs4 import BeautifulSoup
import requests
import re
from PIL import Image
import urllib2 as urllib
import io
import os

# Genearl Parameters
# TODO Set this to 99 for normal run, for debug set 2
Max_Page_Depth = 99 # How many 'Next Page' links to follow

def crawl_style(start_link):
    """ Gets all the urls of images starting from start_link and keeps going
    to the next page"""
    thumbnail_links = [] # place to store all the JPG links
    current_link = start_link
    for count_unused in range(Max_Page_Depth): # A useless counter
        current_page = requests.get(current_link)
        data = current_page.text
        soup = BeautifulSoup(data)
        lcont = soup.find_all(id="listContainer")
        #links = [i.get('href') for i in lcont[0].find_all('a')]
        #r2  = requests.get(domain + links[2])
        links = lcont[0].find_all('a')
        lns2 = [i.find('img') for i in links]
        print 'Current page is ', current_link, ' Current link count is ' \
            ,len(thumbnail_links)
        thumbnail_links += [i.get('src') for i in lns2 if i!=None]
        # all thumbnail pics in one page
        # next get 'next' page link
        # Next Xpath /html/body/div[1]/div/div[2]/div[1]/div[8]/div/a[11]
        # Not using xpath here, found this at http://stackoverflow.com/questions/
        #16992100/going-to-the-next-page-using-python-3-and-beautifulsoup-4
        next_link = soup.find('a', href=True, text=re.compile("Next"))
        if next_link:
            current_link = next_link["href"]
        else:
            return thumbnail_links
    return thumbnail_links

def link_to_PILim(link):
    """ Given a URL of web image, converts it into PIL image"""
    try:
        fd = urllib.urlopen(link)
        image_file = io.BytesIO(fd.read())
        im = Image.open(image_file)
    except UnicodeEncodeError:
        print 'Something wrong with ', link
        return None
    return im

def get_style(style='abstract-art',
              root_path='../Data/Paintings/'):
    """Given a style of painting, downloads all the thumbnails from wikiart
    and saves it in a folder"""
    domain = "http://www.wikiart.org"
    current_link = domain + "/en/paintings-by-style/" + style
    ths =crawl_style(current_link)
    #new_size = (64,64)
    directory = root_path + style
    if not os.path.exists(directory):
        os.makedirs(directory)
    for i in ths:
        im = link_to_PILim(i)
        if im is None:
            continue
        file_name = (re.sub(u"[\\\\,\:,\/]", '_', urllib.url2pathname(i)))[31:]
        # Optional resize im.resize(new_size)
        im.save(directory + '/' + file_name)
    print 'Saved ', len(ths), ' files into dir:', style

def get_all_styles(root_path='../Data/Paintings/'):
    """
    Get many styles of paintings
    """
    # Check if the root path exists, else create it
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    print ' Getting many styles of paintings from internet... '
    styles = ['color-field-painting',
            #'realism',
            'impressionism',
            #'expressionism',
            'surrealism',
            'abstract-art'
              ]
    for particular_style in styles:
        get_style(particular_style, root_path)

def images_to_numpy(input_folder='../Data/Paintings/'):
    """ Takes folder containing folders of images and converts them into data
    suitable for theano classification

    The input_folder is expected to contain different styles of paintings to be
    classified, each subfolder whose name is the style.

    :type input_folder : string
    :param input_folder : path to the folder that contains paintings.

    Returns a tuple of two elements, first is the name of the saved training
    numpy file and second is test file.

    """
    import os
    import glob
    print 'Converting images into Numpy matrix'
    Xl = [] # List that will hold all the (X,y) values
    # Get the subfolders of the input folder
    folders = [i[1] for i in os.walk(input_folder) if i[1]!=[]]
    for painting_style in folders[0]:
        print 'Converting folder ', painting_style
        # painting_style is the name of the sub folders
        for infile in glob.glob(input_folder+ '/' + painting_style \
                                + '/*.jpg'):
            file, ext = os.path.splitext(infile)
            import matplotlib.image as mpimg
            data_im = mpimg.imread(infile)
            #im = Image.open(infile)
            #if resize:
            #    im.thumbnail(new_size, Image.ANTIALIAS)
            #data_im = numpy.array(im)
            flat_d = data_im.flatten() # flatten it
            # Don't think need to normalize, they're not doing it in example
            #normalized = (flat_d.astype(numpy.float32) - 128)/128
            #empty_d = numpy.zeros(new_size[0]*new_size[1]*3,dtype=numpy.int16)
            #empty_d[:flat_d.shape[0]] = flat_d
            # Append index of the folder
            yi = folders[0].index(painting_style)
            Xandy = numpy.hstack((flat_d,yi))
            Xl.append(Xandy)
    from random import shuffle # Shuffle the list
    shuffle(Xl)
    N_train = int(len(Xl)*0.9)
    train_data = numpy.vstack(Xl[:N_train])
    test_data = numpy.vstack(Xl[N_train:])
    print ' Training sample count = ', len(train_data)
    print ' Testing sample count = ', len(test_data)
    print ' Feature dimensionality = ', train_data.shape[1] - 1
    train_filename = input_folder + 'Paintings_train.csv'
    test_filename = input_folder + 'Paintings_test.csv'
    numpy.savetxt(train_filename, train_data,fmt='%d',delimiter=',')
    numpy.savetxt(test_filename, test_data,fmt='%d',delimiter=',')
    return (train_filename, test_filename)

def unpickle_cifar(file):
    """ Loads variable from file"""
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

def visualize_paintings(fn,size_h=64, size_w=64, colors=3):
    """ Function to visualize the saved numpy data file.

    :type fn: string
    :param fn: the numpy file to laod, expected to have ncols*nrows where
               the columns correspond to the samples and rows are features +
               label
    :size_h: height of image, default 64
    :size_w: width of image, default 64
    :colors: 3 for rbg, 1 for greyscale
    """
    all_data = numpy.loadtxt(fn,delimiter=',',dtype=numpy.uint8)
    some_data = all_data[:9] # take only some of them to visualize
    X = some_data[:,:-1]
    y = some_data[:,-1]
    if colors == 1:
        some_data_rs = [i.reshape(size_h,size_w) for i in X]
    else:
        some_data_rs = [i.reshape(size_h,size_w,colors) for i in X]
    import matplotlib.pyplot as plt
    for i in range(len(X)):
        plt.subplot(3,3,i)
        if colors == 1:
            import matplotlib.cm as cm
            plt.imshow(some_data_rs[i],interpolation='None',cmap = cm.Greys_r)
        else:
            plt.imshow(some_data_rs[i],interpolation='None')
        plt.colorbar()
        plt.title(str(y[i]))
    plt.show()

def convert_to_grayscale(numpy_file,size_h, size_w):
    """Given a numpy file, converts it into numpy image matrix file.

    :numpy_file: name of input numpy file.
    :size_h: height of image
    :size_w: width of image
    """
    gray_file = numpy_file[:-4] + '_grey.csv'
    all_data = numpy.loadtxt(numpy_file,delimiter=',',dtype=numpy.uint8)
    X = all_data[:,:-1]
    X_bw = []
    y =all_data[:,-1]
    for i in X:
        rgb = i.reshape(size_h, size_w, 3)
        gray = rgb2gray(rgb)
        X_bw.append(gray.flatten())
    Xandy = numpy.hstack((X_bw,y.reshape(-1,1)))
    numpy.savetxt(gray_file,Xandy,fmt='%d',delimiter=',')
    print 'Converted ', numpy_file, ' to greyscale'


def rgb2gray(rgb):
    """Given integeger array rgb (colors) converts it into greyscale"""
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

if __name__ == '__main__':
    root_path='../data/two_class_full_size/'
    get_all_styles(root_path)
    #images_to_numpy(root_path)
    #visualize_paintings(root_path+ 'Paintings_test.csv')
    #convert_to_grayscale(root_path+ 'big/Paintings_train.csv',64,64)
    #convert_to_grayscale(root_path+ 'big/Paintings_test.csv',64,64)
    #visualize_paintings(root_path+ 'big/Paintings_test_grey.csv',64,64,1)

