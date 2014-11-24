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

def get_y_from_shared(shared_y):
    """Given a theano symbol of shared y gives back normal python var"""
    x = T.vector('x')
    get_y_func = theano.function([], x, givens={x:shared_y})
    return get_y_func()

