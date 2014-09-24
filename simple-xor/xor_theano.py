# -*- coding: utf-8 -*-
import numpy
import time
import os
import sys
import numpy
import theano
import theano.tensor as T
if not sys.path.count("../lib"): sys.path.append("../lib")
import log_debug


def load_data(train_filename, test_filename,split_ratio = [0.8,0.2]):
    """ Loads the data from filename using numpy."""
    traindataset = numpy.loadtxt(train_filename)
    testdataset = numpy.loadtxt(test_filename)
    #Split into train, test and validation set
    N = len(traindataset)
    X = traindataset[:,:-1]
    y = traindataset[:,-1]
    N_train = numpy.round(N*split_ratio[0])
    train_set_x, valid_set_x = X[:N_train], X[N_train:]
    test_set_x = testdataset[:,:-1]
    n_out=1
    test_set_y = testdataset[:,-1].reshape(-1,n_out)
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
    """ Start of main"""
    learning_rate=0.01
    L1_reg=0.00
    L2_reg=0.0001
    n_epochs=100
    train_filename ='data.txt'
    test_filename = 'xor_test.txt'
    batch_size=10
    n_hidden=2
    datasets = load_data(train_filename, test_filename)
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    #pdb.set_trace()
    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size
    n_valid_batches /= batch_size
    n_test_batches /= batch_size
    #classifier = mlp.MLP(rng, )

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    rng = numpy.random.RandomState(1234)
    n_in = train_set_x.get_value(borrow=True).shape[1]
    # construct the MLP class
    #classifier = mlp.MLP(rng=rng, input=x, n_in=n_in,
    #                 n_hidden=n_hidden, n_out=1)

    classifier =log_debug.LogisticRegression(input=x, n_in=n_in, n_out=1)

    cost = classifier.p_y_given_x# \

    train_model = theano.function(inputs=[index], outputs=classifier.ng2,
            givens={
                y: train_set_y[index * batch_size:(index + 1) * batch_size]})
    print train_model(1)

"""
    ###############
    # TRAIN MODEL #
    ###############
    print '... training'

    # early-stopping parameters
    patience = 1000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_params = None
    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()
    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                     (epoch, minibatch_index + 1, n_train_batches,
                      this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                           improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [test_model(i) for i
                                   in xrange(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                    done_looping = True
                    break

    end_time = time.clock()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

print ' End '
"""
