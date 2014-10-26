"""
General data management utility file

Author: Abhishek Rao
"""
import numpy
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import requests
import re
from PIL import Image
import urllib2 as urllib
import io
import os
# TODO Set this to 999 for normal run, for debug set 2
Max_Page_Depth = 999

def crawl_style(start_link):
    """ Gets all the urls of images starting from start_link and keeps going
    to the next page"""
    thumbnail_links = [] # place to store all the JPG links
    current_link = start_link
    for count_unused in range(Max_Page_Depth): # A useless counter
        print 'Current page is ', current_link
        current_page = requests.get(current_link)
        data = current_page.text
        soup = BeautifulSoup(data)
        lcont = soup.find_all(id="listContainer")
        #links = [i.get('href') for i in lcont[0].find_all('a')]
        #r2  = requests.get(domain + links[2])
        links = lcont[0].find_all('a')
        lns2 = [i.find('img') for i in links]
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
              root_path='C:/Users/akr156/Pictures/Paintings/'):
    """Given a style of painting, downloads all the thumbnails from wikiart
    and saves it in a folder"""
    print 'Currently getting style ', style
    domain = "http://www.wikiart.org"
    current_link = domain + "/en/paintings-by-style/" + style
    ths =crawl_style(current_link)
    new_size = (64,64)
    directory = root_path + style
    if not os.path.exists(directory):
        os.makedirs(directory)
    for i in ths:
        im = link_to_PILim(i)
        if im is None:
            continue
        length = len(i)
        file_name = (re.sub(u"[\\\\,\:]", '_', urllib.url2pathname(i)))[31:]
        im.resize(new_size).save(directory + '/' + file_name)

def get_all_styles(root_path='C:/Users/akr156/Pictures/Paintings/'):
    """
    Get many styles of paintings
    """
    print ' Getting many styles of paintings from internet... '
    styles = ['color-field-painting',
            'realism',
            'impressionism',
            'surrealism',
            'abstract-art']
    for particular_style in styles:
        get_style(particular_style, root_path)

def images_to_numpy(input_folder='C:/Users/akr156/Pictures/Paintings/',
                    resize=False):
    """ Takes folder containing folders of images and converts them into data
    suitable for theano classification

    The input_folder is expected to contain different styles of paintings to be
    classified, each subfolder whose name is the style.

    :input_folder (Paintings): A string that is path to the folder that contains
        paintings.
    :resize (False): Bool to decide resize or not
    """
    import os
    import glob
    from PIL import Image
    print 'Converting images into Numpy matrix'
    new_size = (64,64)
    Xl = [] # List that will hold all the (X,y) values
    # Get the subfolders of the input folder
    folders = [i[1] for i in os.walk(input_folder) if i[1]!=[]]
    for painting_style in folders[0]:
        print 'Converting folder ', painting_style
        # painting_style is the name of the sub folders
        for infile in glob.glob(input_folder+ '/' + painting_style \
                                + '/*.jpg'):
            file, ext = os.path.splitext(infile)
            im = Image.open(infile)
            if resize:
                im.thumbnail(new_size, Image.ANTIALIAS)
            data_im = numpy.array(im)
            flat_d = data_im.flatten() # flatten it
            # Don't think need to normalize, they're not doing it in example
            #normalized = (flat_d.astype(numpy.float32) - 128)/128
            empty_d = numpy.zeros(new_size[0]*new_size[1]*3,dtype=numpy.int16)
            empty_d[:flat_d.shape[0]] = flat_d
            # Append index of the folder
            Xandy = numpy.hstack((empty_d, folders[0].index(painting_style)))
            Xl.append(Xandy)
    from random import shuffle # Shuffle the list
    shuffle(Xl)
    N_train = int(len(Xl)*0.9)
    train_data = numpy.vstack(Xl[:N_train])
    test_data = numpy.vstack(Xl[N_train:])
    numpy.savetxt(input_folder + 'Paintings_train.csv',
                  train_data,fmt='%d',delimiter=',')
    numpy.savetxt(input_folder + 'Paintings_test.csv',
                  test_data,fmt='%d',delimiter=',')
    return train_data

# TODO Delete this
#def paint_to_data(fn):
#    """Given a painting jpg file, convert that into numpy array"""
#    from PIL import Image
#    import glob, os
#    size = 128, 128
#    for infile in glob.glob("*.jpg"):
#        file, ext = os.path.splitext(infile)
#        im = Image.open(infile)
#        im.thumbnail(size, Image.ANTIALIAS)
#        im.save(file + ".thumbnail", "JPEG")

def unpickle_cifar(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

def pkl_to_datasets(pickle_files=['cifar-10-batches-py/data_batch_1']
                   ,split_ratio=[0.8,0.1]):
    """
    Converts to theano dataset from pickled file given in
    http://www.cs.toronto.edu/~kriz/cifar.html
    """
    Xl = []
    yl = []
    for fn in pickle_files:
	    all_data_dictionary = unpickle_cifar(fn)
	    Xl.append(all_data_dictionary['data'])
	    yl.append(numpy.array(all_data_dictionary['labels']))
    X = numpy.vstack(Xl)
    y = numpy.concatenate(yl)
    N = X.shape[0]
    N_train = int(N*split_ratio[0])
    N_valid = int(N*split_ratio[1])
    # Split X
    train_set_x = X[:N_train]
    valid_set_x= X[N_train:N_train+N_valid]
    test_set_x = X[N_train+N_valid:]
    # Split y
    train_set_y = y[:N_train]
    valid_set_y= y[N_train:N_train+N_valid]
    test_set_y = y[N_train+N_valid:]

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    print ' Size of train Set x, train set y is ', train_set_x.shape,\
        train_set_y.shape
    print ' Size of valid Set x, valid set y is ', valid_set_x.shape,\
        valid_set_y.shape
    print ' Size of test Set x, test set y is ', test_set_x.shape, \
        test_set_y.shape
    test_set_x, test_set_y = shared_dataset((test_set_x, test_set_y))
    valid_set_x, valid_set_y = shared_dataset((valid_set_x, valid_set_y))
    train_set_x, train_set_y = shared_dataset((train_set_x, train_set_y))
    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval

def load_data(train_filename, test_filename,split_ratio = [0.8,0.2]):
    """ Loads the data from filename using numpy."""
    traindataset = numpy.loadtxt(train_filename,delimiter=",")
    testdataset = numpy.loadtxt(test_filename,delimiter=",")
    #Split into train, test and validation set
    N = len(traindataset)
    X = traindataset[:,:-1]
    y = traindataset[:,-1]
    N_train = numpy.round(N*split_ratio[0])
    train_set_x, valid_set_x = X[:N_train], X[N_train:]
    test_set_x = testdataset[:,:-1]
    test_set_y = testdataset[:,-1]
    train_set_y, valid_set_y= y[:N_train], y[N_train:]
    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    print ' Size of train Set x, train set y is ', train_set_x.shape,\
        train_set_y.shape
    print ' Size of valid Set x, valid set y is ', valid_set_x.shape,\
        valid_set_y.shape
    print ' Size of test Set x, test set y is ', test_set_x.shape, \
        test_set_y.shape
    test_set_x, test_set_y = shared_dataset((test_set_x, test_set_y))
    valid_set_x, valid_set_y = shared_dataset((valid_set_x, valid_set_y))
    train_set_x, train_set_y = shared_dataset((train_set_x, train_set_y))
    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval

if __name__ == '__main__':
    root_path='C:/Users/akr156/Pictures/Paintings/'
    #get_all_styles(root_path)
    images_to_numpy(root_path)
