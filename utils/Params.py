__author__ = 'liushuman'

import numpy
import theano
import theano.tensor as T


class Params(object):

    def __init__(self, if_shared=True):
        self._if_shared = if_shared

    def new(self, value):
        param = numpy.array(value, dtype=theano.config.floatX)

        if self._if_shared:
            param = theano.shared(value=param)

        return param

    def constant(self, size, value=0):
        param = numpy.ones(size, dtype=theano.config.floatX) * value

        if self._if_shared:
            param = theano.shared(value=param)

        return param

    def uniform(self, size, low=-1.0, high=1.0):
        param = numpy.random.uniform(low, high, size)

        if self._if_shared:
            param = theano.shared(value=param)

        return param