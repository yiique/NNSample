__author__ = 'liushuman'

import theano
import theano.tensor as T

import sys
sys.path.append("..")
from utils.Params import Params
from utils.ActivationFunctions import ActivationFunctions


class TestUtils(object):

    def __init__(self):
        self.params = []

    def test_params(self):
        value = [[1, 2, 3]]

        params_new = Params(False).new(value)
        params_new_shared = Params(True).new(value)
        print params_new.dtype, params_new
        print params_new_shared.dtype, params_new_shared.get_value()

        params_constant = Params(False).constant((2, 3))
        params_constant_shared = Params(True).constant((2, 3), value=5)
        print params_constant.dtype, params_constant
        print params_constant_shared.dtype, params_constant_shared.get_value()

        params_uniform_shared = Params(True).uniform((3, 2), low=-0.1, high=0.1)
        print params_uniform_shared.dtype, params_uniform_shared.get_value()

    def test_activation_functions(self):
        func_input = T.matrix()

        sigmoid_result = ActivationFunctions().apply(func_input)
        tanh_result = ActivationFunctions().apply(func_input, activate_type='tanh')

        fn = theano.function([func_input], [sigmoid_result, tanh_result])
        print fn([[-1, 1, 2]])


if __name__ == '__main__':
    test_utils = TestUtils()

    # test_utils.test_params()
    test_utils.test_activation_functions()
