__author__ = 'liushuman'

import numpy as np
import pickle
import theano.tensor as T
import time

from collections import OrderedDict
from numpy import *
from theano import *

# embedding?
# activate function?
# loss function?
# learning rate?
# convergence

# super arguments
#   - if_emb                                y   n
#   - if_bias                               y   n
#   - if_dropout                            TODO
#   - activate_function_for_hidden          TODO
#   - activate_function_for_output          TODO
#   - loss_function                         TODO
#   - learning_rate_method                  TODO
#   - if_momentum                           TODO
#
#   - input_row
#   - input_col & hidden_row
#   - hidden_col & output_row
#   - output_col
#   - convergence_condition                 round_num   total_nll
#
#   - training_file_path
#   - test_file_path
#   - weight_save_path


class RNNModel(object):

    def __init__(self, config_path):
        f_config = open(config_path, "r")
        self.training_file_path = f_config.readline()[:-1]
        self.testing_file_path = f_config.readline()[:-1]
        self.weight_save_path = f_config.readline()[:-1]

        self.if_emb = f_config.readline()[:-1]
        self.if_bias = f_config.readline()[:-1]
        self.if_dropout = f_config.readline()[:-1]
        self.activate_function_for_hidden = f_config.readline()[:-1]
        self.activate_function_for_output = f_config.readline()[:-1]
        self.loss_function = f_config.readline()[:-1]
        self.learning_rate = f_config.readline()[:-1]

        self.emb_row = f_config.readline()[:-1]
        self.emb_col = f_config.readline()[:-1]
        self.input_col = f_config.readline()[:-1]
        self.hidden_col = f_config.readline()[:-1]
        self.output_col = f_config.readline()[:-1]
        self.convergence_condition = f_config.readline()[:-1]
        f_config.close()

        try:
            f_weight = open(self.weight_save_path, 'r')
            self.round_num = pickle.load(f_weight)
            if self.if_emb == 'y':
                self.emb = pickle.load(f_weight)
            self.weight_x = pickle.load(f_weight)
            self.weight_h = pickle.load(f_weight)
            self.weight_o = pickle.load(f_weight)
            self.hidden0 = pickle.load(f_weight)
            if self.if_bias == 'y':
                self.bias_h = pickle.load(f_weight)
                self.bias_o = pickle.load(f_weight)
            f_weight.close()
        except:
            self.init_weight()

        self.params = [self.emb, self.weight_x, self.weight_h, self.weight_o, self.hidden0, self.bias_h, self.bias_o]
        if self.if_emb != 'y':
            self.params = self.params[1:]
        if self.if_bias != 'y':
            self.params = self.params[:-2]

    def init_weight(self):
        self.round_num = 0
        if self.if_emb == 'y':
            self.emb = theano.shared(numpy.random.uniform(-1.0, 1.0, (self.emb_row, self.emb_col)))
        self.weight_x = theano.shared(numpy.random.uniform(-1.0, 1.0, (self.input_col*self.emb_col, self.hidden_col)))
        self.weight_h = theano.shared(numpy.random.uniform(-1.0, 1.0, (self.hidden_col, self.hidden_col)))
        self.weight_o = theano.shared(numpy.random.uniform(-1.0, 1.0, (self.hidden_col, self.output_col)))
        self.hidden0 = theano.shared(numpy.zeros(self.hidden_col))
        if self.if_bias == 'y':
            self.bias_h = theano.shared(numpy.zeros(self.hidden_col))
            self.bias_o = theano.shared(numpy.zeros(self.output_col))

    def save_weight(self):
        f_weight = open(self.weight_save_path, 'w')
        pickle.dump(self.round_num, f_weight)
        if self.if_emb == 'y':
            pickle.dump(self.emb, f_weight)
        pickle.dump(self.weight_x, f_weight)
        pickle.dump(self.weight_h, f_weight)
        pickle.dump(self.weight_o, f_weight)
        pickle.dump(self.hidden0, f_weight)
        if self.if_bias == 'y':
            pickle.dump(self.bias_h, f_weight)
            pickle.dump(self.bias_o, f_weight)
        f_weight.close()

    def generate_functions(self):
        # params
        inputs = T.matrix('input')
        if self.if_emb:
            x = self.emb[inputs].reshape(inputs.shape[0], self.input_col*self.emb_col)
        else:
            x = inputs
        y = T.iscalar('y')
        lr = T.scalar('lr')

        # RNN
        recurrence_dict = {'y_sigmoid_sigmoid': self.recurrence_with_bias_sigmoid_sigmoid,
                           'n_sigmoid_sigmoid': self.recurrence_without_bias_sigmoid_sigmoid,
                           'y_tanh_tanh': self.recurrence_with_bias_tanh_tanh,
                           'n_tanh_tanh': self.recurrence_without_bias_tanh_tanh}
        func_str = self.if_bias + "_" + self.activate_function_for_hidden + "_" + self.activate_function_for_output

        [hidden, output], _ = theano.scan(fn=recurrence_dict[func_str], sequences=x,
                                          outputs_info=[self.hidden0, None], n_steps=x.shape[0])
        p_y_given_x = output[-1]
        y_pred = T.argmax(p_y_given_x)

        # updates
        nll = self.calculate_loss(y, y_pred, p_y_given_x)
        gradients = T.grad(nll, self.params)
        updates = OrderedDict((p, p-lr*g) for p, g in zip(self.params, gradients))

        # functions
        self.theano_train = theano.function(inputs=[inputs, y, lr], outputs=nll, updates=updates)
        if self.if_emb:
            self.theano_normalize = theano.function(inputs=[], updates={self.emb: self.emb/T.sqrt((self.emb**2).sum(axis=1)).dimshuffle(0, 'x')})
        self.theano_classify = theano.function(inputs=[inputs], outputs=y_pred)
        self.theano_encode = theano.function(inputs=[inputs], outputs=hidden[-1])

    def train(self):
        pass

    def test(self):
        pass

    # different recurrence method
    def recurrence_with_bias_sigmoid_sigmoid(self, x_t, h_tm1):
        h_t = T.nnet.sigmoid(T.dot(x_t, self.weight_x) + T.dot(h_tm1, self.weight_h) + self.bias_h)
        o_t = T.nnet.sigmoid(T.dot(h_t, self.weight_o) + self.bias_o)
        return [h_t, o_t]

    def recurrence_without_bias_sigmoid_sigmoid(self, x_t, h_tm1):
        h_t = T.nnet.sigmoid(T.dot(x_t, self.weight_x) + T.dot(h_tm1, self.weight_h))
        o_t = T.nnet.sigmoid(T.dot(h_t, self.weight_o))
        return [h_t, o_t]

    def recurrence_with_bias_tanh_tanh(self, x_t, h_tm1):
        h_t = T.tanh(T.dot(x_t, self.weight_x) + T.dot(h_tm1, self.weight_h) + self.bias_h)
        o_t = T.tanh(T.dot(h_t, self.weight_o) + self.bias_o)
        return [h_t, o_t]

    def recurrence_without_bias_tanh_tanh(self, x_t, h_tm1):
        h_t = T.tanh(T.dot(x_t, self.weight_x) + T.dot(h_tm1, self.weight_h))
        o_t = T.tanh(T.dot(h_t, self.weight_o))
        return [h_t, o_t]

    # different loss functions
    def calculate_loss(self, y, y_pred, p_y_given_x):
        if self.loss_function == '01':
            nll = T.neq(y, y_pred)
        elif self.loss_function == 'log':
            nll = -T.log(p_y_given_x)[y]
        elif self.loss_function == 'square':
            nll = (1-p_y_given_x[y])**2
        return nll

    # dropout
    def dropout_filter(self, updates):
        filter = np.random.binomial(n=1, p=float(self.if_dropout[1:]), size=self.input_col*self.emb_col)
        # TODO


'''

    def train(self):
        print "Training round: ", self.iteration_num
        self.iteration_num += 1

        preprocessor = PreProcessor()

        count = 0
        for sentence in preprocessor.segment:
            sequence = [preprocessor.generate_word_index(x)+1 for x in sentence]

            for i in range(self.min_size, len(sequence)):
                idxs = self.generate_idxs(sequence[:i])
                y = sequence[i] - 1
                # print idxs, y

                nll = self.theano_train(idxs, y, self.learning_rate)
                self.theano_normalize()

                if count % 1000 == 0:
                    print "========"
                    print "training progress: ", count, time.ctime()
                    print "idxs"
                    print idxs
                    print "sequences:"
                    print sequence
                    print ' '.join(sentence[0:i])
                    print "y:", y, preprocess.dict[y], "y_pred:", preprocess.dict[self.theano_classify(idxs)]
                    print "nll at ", count, ":", nll
                count += 1
                #if count == 101:
                #    break
            #if count == 101:
            #    break

        self.closeEvent()

    def generate_idxs(self, sequence):
        idxs = []
        half_cs = self.cs/2

        for i in range(0, len(sequence)):
            idxs_line = [sequence[i]]
            for j in range(0, half_cs):
                if i - (j + 1) >= 0:
                    idxs_line.insert(0, sequence[i - (j+1)])
                else:
                    idxs_line.insert(0, 0)

            for j in range(0, half_cs):
                if i + (j + 1) < len(sequence):
                    idxs_line.append(sequence[i + j + 1])
                else:
                    idxs_line.append(0)

            idxs.append(idxs_line)

        return idxs


if __name__ == "__main__":
    preprocess = PreProcessor()
    rnnencoder = RNNencoder()
    rnnencoder.train()

    #sentence = "The garden is west of the bathroom"
    #idxs = rnnencoder.generate_idxs([preprocess.generate_word_index(x)+1 for x in sentence.split(' ')])
    #print idxs
    #print rnnencoder.theano_encode(idxs)
    '''