__author__ = 'liushuman'

from Preprocessor import Preprocessor

from collections import OrderedDict
from numpy import *

import numpy
import pickle
import theano
import theano.tensor as T

import sys
sys.path.append("../..")
from models.DNNUnit import DNNUnit
from models.GRUUnit import GRUUnit
from models.Pooling import Pooling
from optimal.AdaDelta import AdaDelta


class ReasonerControl(object):

    def __init__(self):
        self.preprocessor = Preprocessor()
        self.cin = 20
        self.cinter = 20
        self.co = 10
        self.dnn_layer = 1
        self.reason_layer = 3

        self.frame_params = []

    def apply(self, _question, _factors, _y):

        for _r in range(self.reason_layer):
            _question = self.interact(_question, _factors)                  # vector 1*20

        dnn_output = DNNUnit((self.co, self.cinter), (1, self.co))
        self.frame_params += dnn_output.params

        _output = T.dot(dnn_output.weight, _question) + dnn_output.bias     # matrix 1*10
        _output = T.nnet.softmax(_output)[0]    # vector 1*10

        _y_pred = T.argmax(_output)
        _nll = -T.log(_output)[_y]
        _grads = T.grad(_nll, self.frame_params)

        return _y_pred, _nll, _grads

    def interact(self, _q_, _f_):

        dnn_q = DNNUnit((self.cinter, self.cin), (1, self.cinter))
        self.frame_params += dnn_q.params

        dnn_f = DNNUnit((self.cinter, self.cin), (1, self.cinter))
        self.frame_params += dnn_f.params

        _q_ = dnn_q.apply(_q_, 'tanh')      # matrix 1*20

        def _interact(__f, __q):
            __f = dnn_f.apply(__f, 'tanh')  # matrix 1*20
            inter_layer = T.concatenate([__q, __f], axis=0)     # matrix 2*20
            return inter_layer

        inter_layers, _ = theano.scan(_interact, sequences=_f_, non_sequences=[_q_])    # matrix 5*2*20
        inter_layers = Pooling().apply(inter_layers, 'mean')                            # matrix 2*20
        inter_layer = Pooling().apply(inter_layers)                                     # vector 1*20
        return inter_layer


if __name__ == "__main__":
    preprocessor = Preprocessor()
    reasoner = ReasonerControl()

    '''test_pair = preprocessor.train_pairs[0]
    question_encode, factors_encode, label = preprocessor.generate_training_data(test_pair)
    print question_encode, factors_encode, label'''

    question = T.dvector('question')
    factors = T.dmatrix('factors')
    y = T.iscalar('y')
    lr = T.scalar('lr')
    # q is vector, factor is matrix 5*20

    _y_pred, _nll, _grads = reasoner.apply(question, factors, y)

    updates = OrderedDict((p, p-lr*g) for p, g in zip(reasoner.frame_params, _grads))

    # fn = theano.function([question, factors, y], [_y_pred, _nll])
    # fn2 = theano.function([question, factors, y], _grads)

    fn_train = theano.function([question, factors, y, lr], [_y_pred, _nll], updates=updates)
    fn_test = theano.function([question, factors], _y_pred)

    iteration_num = 0
    for _ in range(450):
        iteration_num += 1
        print "iteration: ", iteration_num

        total_nll = 0
        precision = 0
        for pair in preprocessor.train_pairs:
            question_encode, factors_encode, label = preprocessor.generate_training_data(pair)
            [y_pred, nll] = fn_train(question_encode, factors_encode, label, 0.08)
            total_nll += nll
            if y_pred == label:
                precision += 1

        print "training nll", total_nll
        print "training precision", str(precision) + "/10000"

        precision = 0
        for pair in preprocessor.test_pairs:
            question_encode, factors_encode, label = preprocessor.generate_training_data(pair)
            y_pred = fn_test(question_encode, factors_encode)
            if y_pred == label:
                precision += 1
        print "testing precision", str(precision) + "/1000"
