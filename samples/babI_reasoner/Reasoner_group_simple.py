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
from models.Pooling import Pooling
from optimal.AdaDelta import AdaDelta


class Reasoner(object):

    def __init__(self):
        self.preprocessor = Preprocessor()
        self.cin = 40
        self.cinter = 20
        self.co = 10
        self.reason_layer = 2
        self.dnn_size = [(40, 1), (32, 1), (32, 1), (20, 1)]

        self.params = []

    def apply(self, _question, _factors, _y):
        # matrix, tensor3, scalar

        for _r in range(self.reason_layer):
            _question = self.interact(_question, _factors)                  # matrix 20*1

        dnn_output = DNNUnit((self.cinter, 1), (self.co, 1))
        self.params += dnn_output.params

        _output = T.dot(dnn_output.weight, _question) + dnn_output.bias     # matrix 10*1
        _output = T.nnet.softmax(_output.T)[0]                              # vector 10*1

        _y_pred = T.argmax(_output)     # scalar
        _nll = -T.log(_output)[_y]
        _grads = T.grad(_nll, self.params)
        return _y_pred, _nll, _grads, _factors, _y

    def interact(self, _q_, _f_):
        # _q_ is matrix and _f_ is tensor3
        dnn_list = []
        for i in range(0, len(self.dnn_size)/2):
            dnn = DNNUnit(self.dnn_size[2*i], self.dnn_size[2*i+1])
            dnn_list.append(dnn)
            self.params += dnn.params

        def _interact(__f, __q):
            # __f and __q are matrixs
            inter_layer = T.concatenate([__q, __f], axis=0)
            for _dnn in dnn_list:
                inter_layer = _dnn.apply(inter_layer)
            return inter_layer        # matrix 20*1

        inter_layers, _ = theano.scan(_interact, sequences=_f_, non_sequences=[_q_])        # tensor3
        inter_layer = Pooling().apply(inter_layers, 'max', axis=0)                          # matrix 20*1

        return inter_layer


if __name__ == "__main__":
    preprocessor = Preprocessor()
    reasoner = Reasoner()

    '''test_pair = preprocessor.train_pairs[0]
    question_encode, factors_encode, label = preprocessor.generate_training_data(test_pair)
    question_encode = [[x] for x in question_encode]
    for i in range(0, len(factors_encode)):
        factors_encode[i] = [[x] for x in factors_encode[i]]
    print question_encode, factors_encode, label'''

    question = T.matrix('question')
    factors = T.tensor3('factors')
    y = T.iscalar('y')
    lr = T.scalar('lr')
    # q is matrix, factor is tensor3 in 5*20*1

    _y_pred, _nll, _grads, _factors, _y = reasoner.apply(question, factors, y)

    updates = OrderedDict((p, p-lr*g) for p, g in zip(reasoner.params, _grads))

    fn = theano.function([question, factors, y, lr], [_y_pred, _nll], updates=updates)
    fn_test = theano.function([question, factors], _y_pred)

    iteration_num = 0
    for _ in range(500):
        iteration_num += 1
        print "iteration: ", iteration_num

        total_nll = 0
        precision = 0
        for pair in preprocessor.train_pairs:
            question_encode, factors_encode, label = preprocessor.generate_training_data(pair)
            question_encode = [[x] for x in question_encode]
            for i in range(0, len(factors_encode)):
                factors_encode[i] = [[x] for x in factors_encode[i]]
            [y_pred, nll] = fn(question_encode, factors_encode, label, 0.08)
            total_nll += nll
            if y_pred == label:
                precision += 1
        print "training nll", total_nll
        print "training precision", str(precision) + "/10000"

        precision = 0
        for pair in preprocessor.test_pairs:
            question_encode, factors_encode, label = preprocessor.generate_training_data(pair)
            question_encode = [[x] for x in question_encode]
            for i in range(0, len(factors_encode)):
                factors_encode[i] = [[x] for x in factors_encode[i]]
            y_pred = fn_test(question_encode, factors_encode)
            if y_pred == label:
                precision += 1
        print "testing precision", str(precision) + "/1000"
