""" File to download bunch of images from a given list of words.
Each word (label) will be downloaded to its own folder"""
import urllib2
import urllib
import simplejson
import os
import time

def download_image_list(urllist, label='default'):
    """ Given a list of url downloads into specified dir"""
    DOWNLOADS_DIR = 'images'
    # For every line in the file
    for url in urllist:
        # Split on the rightmost / and take everything on the right side of that
        # edge case
        if url[-1] == '/':
            url = url[:-1]
        name = url.rsplit('/', 1)[-1]
        # Combine the name and the downloads directory to get the local filename
        directory = os.path.join(DOWNLOADS_DIR, label)
        if not os.path.exists(directory):
            os.makedirs(directory)
        filename = os.path.join(DOWNLOADS_DIR, label, name)
        # Download the file if it does not exist
        if not os.path.isfile(filename):
            urllib.urlretrieve(url, filename)

def download_label(label):
    """Given a search term (label) downloads it from internet"""
    number_of_images = 100
    fetcher = urllib2.build_opener()
    startIndex = 0
    while startIndex < number_of_images:
        searchUrl = "http://ajax.googleapis.com/ajax/services/search/images?v=1.0&q=" + \
            label + "&start=" + str(startIndex)
        f = fetcher.open(searchUrl)
        results = simplejson.load(f)
        if results['responseData'] == None:
            break
        urllist = [i['unescapedUrl'] for i in results['responseData']['results']]
        download_image_list(urllist, label)
        startIndex += len(urllist)
        time.sleep(2)

if __name__ == '__main__':
    labels = ['dog','cat']
    for single_label in labels:
        download_label(single_label)
