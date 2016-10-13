__author__ = 'liushuman'

import theano
import theano.tensor as T


class ActivationFunctions(object):

    def __init__(self):
        self.funcs = {'sigmoid': self._sigmoid,
                      'tanh': self._tanh}

    def apply(self, func_input, activate_type='sigmoid'):
        return self.funcs[activate_type](func_input)

    def _sigmoid(self, func_input):
        func_output = T.nnet.sigmoid(func_input)
        return func_output

    def _tanh(self, func_input):
        func_output = T.tanh(func_input)
        return func_output