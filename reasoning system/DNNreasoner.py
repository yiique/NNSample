__author__ = 'liushuman'

import math
import numpy as np
import pickle
import theano.tensor as T
import time

from collections import OrderedDict
from numpy import *
from theano import *

from preprocess import *
from RNNencoder import *


DI_DICT = {'ee': 1, 'es': 2, 'ew': 3, 'en': 4, 'ss': 5, 'sw': 6, 'sn': 7, 'ww': 8, 'wn': 9, 'nn': 10,
           '1': 'ee', '2': 'es', '3': 'ew', '4': 'en', '5': 'ss', '6': 'sw', '7': 'sn', '8': 'ww', '9': 'wn', '10': 'nn'}


class DNNreasoner(object):
    def __init__(self):
        self.ci = 40    # col of dnn input
        self.ch = 12    # col of dnn hidden
        self.cin = 20   # col of dnn inter result
        self.co = 10    # col of output

        try:
            f = open('DNNweight.save', 'r')
            self.iteration_num = pickle.load(f)
            self.learning_rate_sum = pickle.load(f)
            self.act_learning_rate = pickle.load(f)
            # dnn1
            self.weight_layer11 = pickle.load(f)
            self.bias_layer11 = pickle.load(f)
            self.weight_layer12 = pickle.load(f)
            self.bias_layer12 = pickle.load(f)
            # dnn2
            self.weight_layer21 = pickle.load(f)
            self.bias_layer21 = pickle.load(f)
            self.weight_layer22 = pickle.load(f)
            self.bias_layer22 = pickle.load(f)
            # classify
            self.weight_layer3 = pickle.load(f)
            self.bias_layer3 = pickle.load(f)
            f.close()
        except:
            self.initialize()

        self.params = [self.weight_layer11, self.bias_layer11, self.weight_layer12, self.bias_layer12,
                       self.weight_layer21, self.bias_layer21, self.weight_layer22, self.bias_layer22,
                       self.weight_layer3, self.bias_layer3]

        # params
        question = T.dvector('question')
        factors = T.dmatrix('factors')
        y = T.iscalar('y')
        lr = T.scalar('r')

        # calculate
        def dnn1(f, q):
            input_layer = T.concatenate([q, f])
            hidden_layer = T.nnet.sigmoid(T.dot(self.weight_layer11, input_layer) + self.bias_layer11)
            output_layer = T.nnet.sigmoid(T.dot(self.weight_layer12, hidden_layer) + self.bias_layer12)
            return output_layer

        questions1, _ = theano.scan(fn=dnn1, sequences=factors, non_sequences=[question])
        questions1 = T.mean(questions1, axis=0)

        def dnn2(f, q):
            input_layer = T.concatenate([q, f])
            hidden_layer = T.nnet.sigmoid(T.dot(self.weight_layer21, input_layer) + self.bias_layer21)
            output_layer = T.nnet.sigmoid(T.dot(self.weight_layer22, hidden_layer) + self.bias_layer22)
            return output_layer

        questions2, _ = theano.scan(fn=dnn2, sequences=factors, non_sequences=[questions1])
        questions2 = T.mean(questions2, axis=0)

        answer = T.nnet.softmax(T.dot(self.weight_layer3, questions2) + self.bias_layer3)[0]
        y_pred = T.argmax(answer)

        # update
        nll = -T.log(answer)[y] # + T.log(answer)[y_pred]           This method seems tobe incorrect
        gradients = T.grad(nll, self.params)
        updates = OrderedDict((p, p-lr*g) for p, g in zip(self.params, gradients))

        # funcs
        self.theano_train = theano.function(inputs=[question, factors, y, lr], outputs=nll, updates=updates)
        self.theano_classify = theano.function(inputs=[question, factors], outputs=[answer, y_pred])

        self.test = theano.function(inputs=[question, factors], outputs=[questions1, questions2, y_pred])

    def initialize(self):
        self.iteration_num = 0
        self.learning_rate_sum = 0
        self.act_learning_rate = 1
        # dnn1
        self.weight_layer11 = theano.shared(numpy.random.uniform(-1.0, 1.0, (self.ch, self.ci)))
        self.bias_layer11 = theano.shared(numpy.zeros(self.ch, dtype=theano.config.floatX))
        self.weight_layer12 = theano.shared(numpy.random.uniform(-1.0, 1.0, (self.cin, self.ch)))
        self.bias_layer12 = theano.shared(numpy.zeros(self.cin, dtype=theano.config.floatX))
        # dnn2
        self.weight_layer21 = theano.shared(numpy.random.uniform(-1.0, 1.0, (self.ch, self.ci)))
        self.bias_layer21 = theano.shared(numpy.zeros(self.ch, dtype=theano.config.floatX))
        self.weight_layer22 = theano.shared(numpy.random.uniform(-1.0, 1.0, (self.cin, self.ch)))
        self.bias_layer22 = theano.shared(numpy.zeros(self.cin, dtype=theano.config.floatX))
        # classify
        self.weight_layer3 = theano.shared(numpy.random.uniform(-1.0, 1.0, (self.co, self.cin)))
        self.bias_layer3 = theano.shared(numpy.zeros(self.co, dtype=theano.config.floatX))

    def closeEvent(self):
        f = open('DNNweight.save', 'w')
        pickle.dump(self.iteration_num, f)
        pickle.dump(self.learning_rate_sum, f)
        pickle.dump(self.act_learning_rate, f)
        pickle.dump(self.weight_layer11, f)
        pickle.dump(self.bias_layer11, f)
        pickle.dump(self.weight_layer12, f)
        pickle.dump(self.bias_layer12, f)
        pickle.dump(self.weight_layer21, f)
        pickle.dump(self.bias_layer21, f)
        pickle.dump(self.weight_layer22, f)
        pickle.dump(self.bias_layer22, f)
        pickle.dump(self.weight_layer3, f)
        pickle.dump(self.bias_layer3, f)
        f.close()

    def train(self):
        self.iteration_num += 1
        print "Training round: ", self.iteration_num

        preprocess = PreProcess()
        rnnencoder = RNNencoder()
        total_nll = 0

        count = 0
        for QApair in preprocess.QApairs:
            # lr
            self.learning_rate_sum += self.act_learning_rate**2
            self.act_learning_rate /= math.sqrt(self.learning_rate_sum/self.iteration_num + 0.00001)

            question_encode, factors_encode, y = self.generate_training_data(QApair, preprocess, rnnencoder)
            nll = self.theano_train(question_encode, factors_encode, y, self.act_learning_rate)
            total_nll += nll

            if count % 400 == 0:
                print "=========================="
                print "example:", y, self.index2di(y+1) #question_encode, factors_encode,
                print "training progress: ", count, time.ctime()
                print "lr:",  self.act_learning_rate
                print "nll:", nll

                answer_list = self.theano_classify(question_encode, factors_encode)
                print "classify:", answer_list, self.index2di(answer_list[1]+1)
            # else:
            #     break
            count += 1

        self.closeEvent()
        return total_nll

    def generate_training_data(self, QApair, preprocess, rnnencoder):
        question = QApair["q"]
        factors = QApair["f"]

        question_encode = []
        factors_encode = []

        for word in preprocess.dictionary:
            if word in question:
                question_encode.append(1)
            else:
                question_encode.append(0)

        for factor in factors:
            single_factor = []
            for word in preprocess.dictionary:
                if word in factor:
                    single_factor.append(1)
                else:
                    single_factor.append(0)
            factors_encode.append(single_factor)

        y = self.di2index(QApair["a"])

        return question_encode, factors_encode, y

    def di2index(self, di_list):
        di_order = ['e', 's', 'w', 'n']

        di1 = di_list[0]
        di2 = di_list[1]

        if di_order.index(di1) < di_order.index(di2):
            return DI_DICT[di1+di2] - 1
        else:
            return DI_DICT[di2+di1] - 1

    def index2di(self, index):
        di = DI_DICT[str(index)]

        return [di[0], di[1]]


if __name__ == "__main__":
    # preprocess = PreProcess()
    # rnnencoder = RNNencoder()
    dnnreasoner = DNNreasoner()
    total_nll = 20000

    while total_nll >= 15000:
        total_nll = dnnreasoner.train()
        print "########"
        print "total_nll: ", total_nll
        print "########"

    '''
    question = "How do you go from the kitchen to the garden"
    sentences = ["The office is east of the hallway",
                 "The kitchen is north of the office",
                 "The garden is west of the bedroom",
                 "The office is west of the garden",
                 "The bathroom is north of the garden"]

    question_idxs = rnnencoder.generate_idxs([preprocess.generate_word_index(x)+1 for x in question.split(' ')])
    sentences_idxs = []
    for sentence in sentences:
        sentences_idxs.append(rnnencoder.generate_idxs([preprocess.generate_word_index(x)+1 for x in sentence.split(' ')]))

    question_encode = rnnencoder.theano_encode(question_idxs)[-1]
    sentences_encode = []
    for sentence_idxs in sentences_idxs:
        sentences_encode.append(rnnencoder.theano_encode(sentence_idxs)[-1])

    print dnnreasoner.theano_train(question_encode, sentences_encode, 10, dnnreasoner.learning_rate)'''