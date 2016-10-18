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


class ReasonerControl(object):

    def __init__(self):
        self.preprocessor = Preprocessor()
        self.cin = 40
        self.cinter = 20
        self.co = 10
        self.dnn_layer = 1
        self.reason_layer = 3

        self.control_weight1 = theano.shared(numpy.random.uniform(-1.0, 1.0, (self.cinter, self.cin)))
        self.control_bias1 = theano.shared(numpy.zeros((1, self.cinter), dtype=theano.config.floatX))
        self.control_weight2 = theano.shared(numpy.random.uniform(-1.0, 1.0, (self.cinter, self.cin)))
        self.control_bias2 = theano.shared(numpy.zeros((1, self.cinter), dtype=theano.config.floatX))
        self.control_weight3 = theano.shared(numpy.random.uniform(-1.0, 1.0, (self.cinter, self.cin)))
        self.control_bias3 = theano.shared(numpy.zeros((1, self.cinter), dtype=theano.config.floatX))
        self.control_weight4 = theano.shared(numpy.random.uniform(-1.0, 1.0, (self.co, self.cinter)))
        self.control_bias4 = theano.shared(numpy.zeros((1, self.co), dtype=theano.config.floatX))

        self.frame_params = []
        self.control_params = [self.control_weight1, self.control_bias1, self.control_weight2, self.control_bias2,
                               self.control_weight3, self.control_bias3, self.control_weight4, self.control_bias4]

    def apply(self, _question_f, _question_c, _factors, _y):

        for _r in range(self.reason_layer):
            # input: vector, vector, matrix, layer
            # output: vector, vector
            _question_f, _question_c = self.interact(_question_f, _question_c, _factors, _r)

        dnn_output = DNNUnit((self.co, self.cinter), (1, self.co))
        self.frame_params += dnn_output.params
        control_weight = self.control_params[-2]
        control_bias = self.control_params[-1]
        # control_weight.set_value(dnn_output.params[0].get_value())
        # control_bias.set_value(dnn_output.params[1].get_value())

        _output_frame = T.dot(dnn_output.weight, _question_f) + dnn_output.bias    # vector 1*10
        _output_control = T.dot(control_weight, _question_c) + control_bias[0]  # vector 1*10

        _output_frame = T.nnet.softmax(_output_frame)[0]        # vector 1*10
        _output_control = T.nnet.softmax(_output_control)[0]    # vector 1*10

        _y_pred_frame = T.argmax(_output_frame)
        _y_pred_control = T.argmax(_output_control)

        _nll_frame = -T.log(_output_frame)[_y]
        _nll_control = -T.log(_output_control)[_y]

        _grads_frame = T.grad(_nll_frame, self.frame_params)
        _grads_control = T.grad(_nll_control, self.control_params)
        return _y_pred_frame, _y_pred_control, _nll_frame, _nll_control, _grads_frame, _grads_control, _factors, _y

    def interact(self, _q_f, _q_c, _f_, _r_):

        dnn = DNNUnit((self.cinter, self.cin), (1, self.cinter))
        self.frame_params += dnn.params
        control_weight = self.control_params[2*_r_]
        control_bias = self.control_params[2*_r_ + 1]
        # control_weight.set_value(dnn.params[0].get_value())
        # control_bias.set_value(dnn.params[1].get_value())

        def _interact_frame(__f, __q):
            input_layer = T.concatenate([__q, __f])
            frame_inter_layer = dnn.apply(input_layer)
            return frame_inter_layer        # matrix

        def _interact_control(__f, __q):
            input_layer = T.concatenate([__q, __f])
            control_inter_layer = T.nnet.sigmoid(T.dot(control_weight, input_layer) + control_bias)
            return control_inter_layer      # matrix

        frame_inter_layers, _ = theano.scan(_interact_frame, sequences=_f_, non_sequences=[_q_f])       # matrix 5*1*20
        frame_inter_layer = Pooling().apply(frame_inter_layers, 'mean')     # matrix 1*20

        control_inter_layers, _ = theano.scan(_interact_control, sequences=_f_, non_sequences=[_q_c])   # matrix 5*1*20
        control_inter_layer = T.mean(control_inter_layers, axis=0)          # matrix 1*20

        return frame_inter_layer[0], control_inter_layer[0]


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

    _y_pred_frame, _y_pred_control, _nll_frame, _nll_control, _grads_frame, _grads_control, _factors, _y \
        = reasoner.apply(question, question, factors, y)

    updates_frame = OrderedDict((p, p-lr*g) for p, g in zip(reasoner.frame_params, _grads_frame))
    updates_control = OrderedDict((p, p-lr*g) for p, g in zip(reasoner.control_params, _grads_control))

    # ====test====

    # fn = theano.function([question, factors, y], [_y_pred_frame, _y_pred_control, _factors, _y])
    # fn_grad_f = theano.function([question, factors, y], _grads_frame)
    # fn_grad_c = theano.function([question, factors, y], _grads_control)

    # result = fn(question_encode, factors_encode, label)
    # result_grad_frame = fn_grad_f(question_encode, factors_encode, label)
    # result_grad_control = fn_grad_c(question_encode, factors_encode, label)

    # print params
    '''for p_f, p_c in zip(reasoner.frame_params, reasoner.control_params):
        print p_f.get_value()
        print '----------------'
        print p_c.get_value()
        print "================"'''

    # print grads
    '''for r_f, r_c in zip(result_grad_frame, result_grad_control):
        print r_f
        print '----------------'
        print r_c
        print "================"'''

    # print updates
    '''for p_f, r_f, u_f in zip(reasoner.frame_params, result_grad_frame, updates_frame):
        print p_f.get_value()
        print r_f
        print u_f.get_value()'''

    fn_frame = theano.function([question, factors, y, lr], [_y_pred_frame, _nll_frame], updates=updates_frame)
    fn_control = theano.function([question, factors, y, lr], [_y_pred_control, _nll_control], updates=updates_control)
    fn_frame_test = theano.function([question, factors], _y_pred_frame)
    fn_control_test = theano.function([question, factors], _y_pred_control)

    '''for p_f, p_c in zip(reasoner.frame_params, reasoner.control_params):
        print p_f.get_value()
        print '----------------'
        print p_c.get_value()
        print "================"

    result_frame = fn_frame(question_encode, factors_encode, label)
    result_control = fn_control(question_encode, factors_encode, label)

    print result_frame
    print result_control

    for p_f, p_c in zip(reasoner.frame_params, reasoner.control_params):
        print p_f.get_value()
        print '----------------'
        print p_c.get_value()
        print '----------------'
        print p_f.get_value() == p_c.get_value()
        print "================"'''

    iteration_num = 0
    for _ in range(600):
        iteration_num += 1
        print "iteration: ", iteration_num

        total_nll_frame = 0
        total_nll_control = 0
        precision_frame = 0
        precision_control = 0
        for pair in preprocessor.train_pairs:
            question_encode, factors_encode, label = preprocessor.generate_training_data(pair)
            [y_pred_frame, nll_frame] = fn_frame(question_encode, factors_encode, label, 0.08)
            [y_pred_control, nll_control] = fn_control(question_encode, factors_encode, label, 0.08)
            total_nll_frame += nll_frame
            total_nll_control += nll_control
            if y_pred_frame == label:
                precision_frame += 1
            if y_pred_control == label:
                precision_control += 1
        print "training nll frame", total_nll_frame
        print "training nll control", total_nll_control
        print "training precision frame", str(precision_frame) + "/10000"
        print "training precision control", str(precision_control) + '/10000'

        precision_frame = 0
        precision_control = 0
        for pair in preprocessor.test_pairs:
            question_encode, factors_encode, label = preprocessor.generate_training_data(pair)
            y_pred_frame = fn_frame_test(question_encode, factors_encode)
            y_pred_control = fn_control_test(question_encode, factors_encode)
            if y_pred_frame == label:
                precision_frame += 1
            if y_pred_control == label:
                precision_control += 1
        print "testing precision frame", str(precision_frame) + "/1000"
        print "testing precision control", str(precision_control) + '/1000'
