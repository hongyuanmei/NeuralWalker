import pickle
import time
import numpy
import theano
from theano import sandbox
import theano.tensor as tensor
import os
import scipy.io
from collections import defaultdict
from theano.tensor.shared_randomstreams import RandomStreams

dtype=theano.config.floatX

def sample_weights(nrow, ncol):
    bound = (numpy.sqrt(6.0) / numpy.sqrt(nrow+ncol) ) * 1.0
    # nrow -- # of prev layer units, ncol -- # of this layer units
# this is form Bengio's 2010 paper
    values = numpy.random.uniform(
        low=-bound, high=bound, size=(nrow, ncol)
    )
    return numpy.cast[dtype](values)
