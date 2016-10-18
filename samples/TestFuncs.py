__author__ = 'liushuman'

import theano
import theano.tensor as T

import sys
sys.path.append("..")
from models.DNNUnit import DNNUnit
from models.GRUUnit import GRUUnit
from models.Pooling import Pooling
from optimal.AdaDelta import AdaDelta
from utils.Params import Params


class TestFuncs(object):

    def __init__(self):
        self._func_input = None
        self._func_output = None
        self.params = []

    def test_DNNUnit(self, func_input, weight_size, bias_size):
        self._func_input = T.vector('_func_input')

        dnn_unit = DNNUnit(weight_size, bias_size)
        self.params += dnn_unit.params

        self._func_output = dnn_unit.apply(self._func_input)

        self._theano_fn = theano.function(inputs=[self._func_input], outputs=self._func_output)
        return self._theano_fn(func_input)

    def test_pooling(self, func_input):
        self._func_input = T.matrix('_func_input')
        self._func_output = Pooling().apply(self._func_input, 'mean')

        self._theano_fn = theano.function(inputs=[self._func_input], outputs=self._func_output)
        return self._theano_fn(func_input)

    def test_ada_delta(self, func_input, func_y):
        x_sim = T.matrix()
        y = T.scalar()
        weight_x = Params().uniform((1, 3))
        y_pred = T.dot(weight_x, x_sim)
        params = [weight_x]

        # Note: case the params of ada delta have no business with the grad, it should not be add into the params list
        ada_delta = AdaDelta(params)

        nll = T.sum(y_pred) - y
        grad = T.grad(nll, params)
        updates = ada_delta.apply(params, grad)

        self._theano_fn = theano.function(inputs=[x_sim, y],
                                          outputs=[weight_x, y_pred, nll, grad[0], ada_delta._gradients_accumulate[0]],
                                          updates=updates)
        self._theano_print = theano.function(inputs=[], outputs=[weight_x, ada_delta._gradients_accumulate[0]])

        return self._theano_fn(func_input, func_y) + self._theano_print()

    # def test_reset_gate(self, wx_size, wh_size, bias_size):

    def test_GRU(self, func_input, wx_size, wh_size, h_size, bias_size):
        self._func_input = T.matrix('_func_input')

        grn_unit = GRUUnit(wx_size, wh_size, h_size, bias_size)
        self.params += grn_unit.params

        self._func_output = grn_unit.apply(self._func_input)
        self._theano_fn = theano.function(inputs=[self._func_input], outputs=self._func_output)

        return self._theano_fn(func_input)


if __name__ == '__main__':
    test_funcs = TestFuncs()

    # print test_funcs.test_DNNUnit([1, 2, 3, 4], (2, 4), (1, 2))

    # print test_funcs.test_pooling([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    # output = test_funcs.test_ada_delta([[1], [1], [1]], 1.5)
    # for _ in output:
    #     print _

    output = test_funcs.test_GRU([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7]], (3, 5), (3, 3), (1, 3), (1, 3))
    print output