import os
import sys
import time
import numpy
if not sys.path.count("../../lib"): sys.path.append("../../lib")
from utils_abhi import load_data, get_y_from_shared
from SdA2 import SdA2

# change current working directory to the directory of the file
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

def test_sda(
        train_file_name ='../data/Paintings/two_class/Paintings_train.csv',
        test_file_name ='../data/Paintings/two_class/Paintings_test.csv',
        finetune_lr=0.1,
        pretraining_epochs=15,
        pretrain_lr=0.001,
        training_epochs=1000,
    batch_size=10):
    """
    Demonstrates how to train and test a stochastic dN_train+N_validenoising
    autoencoder.

    This is demonstrated on MNIST.

    :train_file_name: string
    :param train_file_name: Input numpy training file

    :test_file_name: string
    :param test_file_name: Input numpy testing file

    :type learning_rate: float
    :param learning_rate: learning rate used in the finetune stage
    (factor for the stochastic gradient)

    :type pretraining_epochs: int
    :param pretraining_epochs: number of epoch to do pretraining

    :type pretrain_lr: float
    :param pretrain_lr: learning rate to be used during pre-training

    :type n_iter: int
    :param n_iter: maximal number of iterations ot run the optimizer

    Returns tuple of best validataion loss and test score
    """
    # Change here
    datasets = load_data(train_file_name, test_file_name)
    # End of change
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size

    n_ins = train_set_x.get_value(borrow=True).shape[1]
    n_outs = 2
    # numpy random generator
    numpy_rng = numpy.random.RandomState(89677)
    print '... building the model'
    # construct the stacked denoising autoencoder class
    sda = SdA2(numpy_rng=numpy_rng, n_ins= n_ins,
                hidden_layers_sizes=[1000, 1000, 1000],
                n_outs=n_outs, corruption_levels=[.1,.2,.3])

    #Checking precision
    # get python variable from theano variable
    test_y_pyvar = get_y_from_shared(test_set_y)
    test_x_pyvar = test_set_x.get_value()
    print 'The precision is ', sda.precision(test_x_pyvar, test_y_pyvar)

    #########################
    # PRETRAINING THE MODEL #
    #########################
    print '... getting the pretraining functions'
    pretraining_fns = sda.pretraining_functions(train_set_x=train_set_x,
                                                batch_size=batch_size)

    print '... pre-training the model'
    start_time = time.clock()
    ## Pre-train layer-wise
    corruption_levels = [.1, .2, .3]
    for i in xrange(sda.n_layers):
        # go through pretraining epochs
        for epoch in xrange(pretraining_epochs):
            # go through the training set
            c = []
            for batch_index in xrange(n_train_batches):
                c.append(pretraining_fns[i](index=batch_index,
                            corruption=corruption_levels[i],
                            lr=pretrain_lr))
            print 'Pre-training layer %i, epoch %d, cost ' % (i, epoch),
            print numpy.mean(c)

    end_time = time.clock()

    print >> sys.stderr, ('The pretraining code for file ' +
                            os.path.split(__file__)[1] +
                            ' ran for %.2fm' % ((end_time - start_time) / 60.))

    ########################
    # FINETUNING THE MODEL #
    ########################

    # get the training, validation and testing function for the model
    print '... getting the finetuning functions'
    train_fn, validate_model, test_model = sda.build_finetune_functions(
                datasets=datasets, batch_size=batch_size,
                learning_rate=finetune_lr)

    print '... finetunning the model'
    # early-stopping parameters
    patience = 10 * n_train_batches  # look as this many examples regardless
    patience_increase = 2.  # wait this much longer when a new best is
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
    test_score = 0.
    start_time = time.clock()

    done_looping = False
    epoch = 0

    while (epoch < training_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):
            minibatch_avg_cost = train_fn(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                validation_losses = validate_model()
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                        (epoch, minibatch_index + 1, n_train_batches,
                        this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if (this_validation_loss < best_validation_loss *
                        improvement_threshold):
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = test_model()
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                            'best model %f %%') %
                            (epoch, minibatch_index + 1, n_train_batches,
                            test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print(('Optimization complete with best validation score of %f %%,'
            'with test performance %f %%') %
                    (best_validation_loss * 100., test_score * 100.))
    print >> sys.stderr, ('The training code for file ' +
                            os.path.split(__file__)[1] +
                            ' ran for %.2fm' % ((end_time - start_time) / 60.))
    #Checking precision
    # get python variable from theano variable
    test_y_pyvar = get_y_from_shared(test_set_y)
    test_x_pyvar = test_set_x.get_value()
    print 'The precision is ', sda.precision(test_x_pyvar, test_y_pyvar)
    print 'The recall is ', sda.recall(test_x_pyvar, test_y_pyvar)
    # End
    return sda


if __name__ == '__main__':
    sda = test_sda()

