"""
Code to get paintings from web and convert it into numpy file.

Date: 26 Oct 2014
Author: Abhishek Rao

The general algorithm is as follows.

1. List all styles you want
2. For each style create a folder
3. Crawl that style and get list of links for images
4. For each link
5. Download the links using urllib and convert to PIL image, resize
   and write to JPG file.
6. For each image convert to numpy array and append the folder index to create
   a matrix where the rows are samples, columns = features + label,
   hence the number of columns = number of features + 1. Save this numpy matrix
   as csv file, create two, one for train, and one for test.
"""
import numpy
import theano
import theano.tensor as T
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
              root_path='../Data/Paintings/'):
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
        file_name = (re.sub(u"[\\\\,\:,\/]", '_', urllib.url2pathname(i)))[31:]
        im.resize(new_size).save(directory + '/' + file_name)

def get_all_styles(root_path='../Data/Paintings/'):
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

def images_to_numpy(input_folder='../Data/Paintings/',
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

def unpickle_cifar(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict


if __name__ == '__main__':
    root_path='../Data/Paintings/'
    get_all_styles(root_path)
    images_to_numpy(root_path)
