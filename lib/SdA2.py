"""
Extra functions added to sda, like predict.

Author: Abhishek Rao
Date: 11 Nov 2014

Subclass SdA2 created out of SdA
"""
import numpy
import theano
import SdA
from sklearn.metrics import precision_score, recall_score


class SdA2(SdA.SdA):
    """Stacked denoising auto-encoder class (SdA) with exras like predict.
    """

    def __init__(self, numpy_rng, theano_rng=None, n_ins=784,
                 hidden_layers_sizes=[500, 500], n_outs=10,
                 corruption_levels=[0.1, 0.1]):
        """ This just calls it's parent class init.

        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: numpy random number generator used to draw initial
                    weights
        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator; if None is given one is
                           generated based on a seed drawn from `rng`

        :type n_ins: int
        :param n_ins: dimension of the input to the sdA

        :type n_layers_sizes: list of ints
        :param n_layers_sizes: intermediate layers size, must contain
                               at least one value

        :type n_outs: int
        :param n_outs: dimension of the output of the network

        :type corruption_levels: list of float
        :param corruption_levels: amount of corruption to use for each
                                  layer
        """
        super(SdA2,self).__init__(numpy_rng, theano_rng,
                                  n_ins,hidden_layers_sizes, n_outs,
                                  corruption_levels)

    def predict(self,X):
        """ Predict new value based on the parameters"""
        predict_func = theano.function(inputs=[self.x],
                outputs=self.logLayer.y_pred)
                #outputs=self.logLayer.y_pred,givens={x:X})
        return predict_func(X)

    def precision(self,X,y):
        """ Given input X and y, predicts the output ypred and compares it with
        y and uses sklear precision calculator utility to find the precision of
        prediction"""
        y_pred = self.predict(X)
        return precision_score(y, y_pred)

    def recall(self,X,y):
        """ Given input X and y, predicts the output ypred and compares it with
        y and uses sklear recall calculator utility to find the recall of
        prediction"""
        y_pred = self.predict(X)
        return recall_score(y, y_pred)

if __name__ == '__main__':
    # numpy random generator
    numpy_rng = numpy.random.RandomState(89677)
    # creating the classifier
    sda = SdA2(numpy_rng=numpy_rng, n_ins=2,
            hidden_layers_sizes=[2],
            n_outs=2)
    # getting some test values
    X = numpy.random.rand(3,2)
    # prediction before training
    yp1 = sda.predict(X)
    y = [0, 1, 1]
    print sda.precision(X,y)
    # training the classifier
    #SdA.test_SdA(classifier=sda,training_epochs=10)
    #yp2 = sda.predict(sda.x, X)
