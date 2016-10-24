__author__ = 'liushuman'

import theano
import theano.tensor as T

import sys
sys.path.append("..")
from models.DNNUnit import DNNUnit
from models.Embedding import Embedding
from models.GRUUnit import GRUUnit, ResetGate, UpdateGate
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

    def test_emb(self, func_input, emb_size):
        self._func_input = T.imatrix('_func_input')
        emb = Embedding(emb_size)
        self._func_output = emb.apply(self._func_input)

        self._theano_fn = theano.function(inputs=[self._func_input], outputs=self._func_output)
        return self._theano_fn(func_input)

    def test_gates(self, xt, htm1, wx_size, wh_size, bias_size, context):
        tensor_xt = T.vector()
        tensor_htm1 = T.vector()

        reset_gate = ResetGate(wx_size, wh_size, bias_size, context)
        update_gate = UpdateGate(wx_size, wh_size, bias_size, context)

        tensor_output_reset = reset_gate.apply(tensor_xt, tensor_htm1)
        tensor_output_update = update_gate.apply(tensor_xt, tensor_htm1)

        fn = theano.function([tensor_xt, tensor_htm1], [tensor_output_reset, tensor_output_update])
        return fn(xt, htm1)

    def test_GRU(self, indexs, emb_size, wx_size, wh_size, h_size, bias_size, context, wm_size):
        tensor_indexs = T.imatrix()
        tensor_htm1 = T.vector()

        emb = Embedding(emb_size)
        tensor_xt = emb.apply(tensor_indexs)

        gru_unit = GRUUnit(wx_size, wh_size, h_size, bias_size, context)
        tensor_output_apply = gru_unit.apply(tensor_xt)
        fn_apply = theano.function([tensor_indexs], tensor_output_apply)
        output_apply = fn_apply(indexs)

        # gru_unit_decoder = GRUUnit(wx_size, wh_size, h_size, bias_size, )
        # tensor_output_mergeout = gru_unit_decoder.merge_out(tensor_xt, (2, 13))
        gru_unit_decoder = GRUUnit(wx_size, wh_size, h_size, bias_size, output_apply[-1])
        tensor_output_mergeout = gru_unit_decoder.merge_out(tensor_xt, wm_size)
        fn_merge = theano.function([tensor_indexs], tensor_output_mergeout)
        output_merge = fn_merge(indexs)

        return [output_apply, output_merge]


if __name__ == '__main__':
    test_funcs = TestFuncs()

    # print test_funcs.test_DNNUnit([1, 2, 3, 4], (2, 4), (1, 2))

    # print test_funcs.test_pooling([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    # output = test_funcs.test_ada_delta([[1], [1], [1]], 1.5)
    # for _ in output:
    #     print _

    # print test_funcs.test_emb([[1], [2], [3]], (5, 10))     # matrix 3*10

    print test_funcs.test_gates([1, 1, 1, 1, 1], [0, 0, 0], (3, 5), (3, 3), (1, 3), [[1, 1, 1]])

    output = test_funcs.test_GRU([[1], [2], [3]], (5, 10), (3, 10), (3, 3), (1, 3), (1, 3), [[1, 1, 1]], (2, 16))
    for _ in output:
        print _