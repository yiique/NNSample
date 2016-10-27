__author__ = 'liushuman'

import theano
import theano.tensor as T

import sys
sys.path.append("..")
from utils.ActivationFunctions import ActivationFunctions
from utils.Params import Params


class DNNUnit(object):

    def __init__(self, input_size, output_size, if_bias=True, activate_type='sigmoid'):
        if input_size[1] != output_size[1]:
            print "WARNING IN DNNUNIT: YOUR INPUT COLUMNS IS NOT EQUAL WITH OUTPUT COLUMNS!"

        self._if_bias = if_bias
        self._activate_type = activate_type

        self.weight = Params().uniform((output_size[0], input_size[0]))
        self.params = [self.weight]
        if self._if_bias:
            self.bias = Params().constant(output_size)
            self.params += [self.bias]

    def apply(self, unit_input):
        unit_output = T.dot(self.weight, unit_input)
        if self._if_bias:
            unit_output += self.bias

        unit_output = ActivationFunctions().apply(unit_output)

        return unit_output
