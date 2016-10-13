__author__ = 'liushuman'

import theano
import theano.tensor as T


class Pooling(object):

    def __init__(self):
        self.funcs = {'max': self._max,
                      'mean': self._mean}

    def apply(self, func_input, pooling_type='mean', axis=0):
        return self.funcs[pooling_type](func_input, axis)

    def _max(self, func_input, axis):
        func_output = T.max(func_input, axis)
        return func_output

    def _mean(self, func_input, axis):
        func_output = T.mean(func_input, axis)
        return func_output