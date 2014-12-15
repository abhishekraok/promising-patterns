"""
General data management utility file

Author: Abhishek Rao
"""
import numpy
import theano
import theano.tensor as T

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
    traindataset = numpy.loadtxt(train_filename, dtype=numpy.int16,
                                 delimiter=",")
    testdataset = numpy.loadtxt(test_filename, dtype=numpy.int16,
                                delimiter=",")
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

def get_y_from_shared(shared_y):
    """Given a theano symbol of shared y gives back normal python var"""
    x = T.vector('x')
    get_y_func = theano.function([], x, givens={x:shared_y})
    return get_y_func()

def rpca(numpy_file='../data/Paintings/two_class/Paintings_train.csv'):
    """ Performs randomized PCA on given numpy file.

    Given a numpy file of n-rows and n-cols, where the last column is
    the label and rest are features,n-rows are the samples.

    :type numpy_file: string
    :param numpy_file: The file name of numpy file to be analyzed.
    """
    import numpy as np
    import matplotlib.pyplot as pl
    import pandas as pd
    from sklearn.decomposition import RandomizedPCA

    all_data = np.loadtxt(numpy_file,delimiter=',')
    data = all_data[:,:-1]
    y = all_data[:,-1]
    pca = RandomizedPCA(n_components=2)
    X = pca.fit_transform(data)
    df = pd.DataFrame({"x": X[:, 0], "y": X[:, 1],\
                    "label":np.where(y==1, "realism", "abstract")})
    colors = ["red", "yellow"]
    for label, color in zip(df['label'].unique(), colors):
        mask = df['label']==label
        pl.scatter(df[mask]['x'], df[mask]['y'], c=color, label=label)
    pl.legend()
    pl.title('Randomized PCA analysis')
    pl.show()

def numpy_to_pkl_theano(train_filename, test_filename,
                        pkl_filename=None):
    """Given a numpy data file converts into pickle file suitable for
    use with theano example.

    :param train_filename: string, filename of numpy. Expected to have
    samples by rows and features by columns.
    :param pkl_filename: string (optional), filename of output Cpickle file.
    Default is same as train_filename

    Returns:
        The string name of file saved.
    """
    if pkl_filename == None:
        pkl_filename = train_filename[:-4] + '.pkl.gz'
    rval = load_data(train_filename, test_filename)
    import gzip
    import cPickle
    f = gzip.open(pkl_filename, 'wb')
    cPickle.dump(rval, f)
    f.close()
    print 'File ', pkl_filename, ' created'
    return pkl_filename



