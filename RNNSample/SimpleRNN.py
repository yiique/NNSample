__author__ = 'liushuman'

import pickle
import theano.tensor as T
import time

from collections import OrderedDict
from numpy import *
from theano import *

from SimpleRNNPreprocessor import *


class RNNencoder(object):

    def __init__(self):
        self.cs = 5             # word window context size
        self.ne = 20002         # number of emb & input         dict_size + 1
        self.nh = 100           # number of hidden
        self.de = 100           # dimension of emb
        self.nc = 20000         # number of class & output
        self.learning_rate = 0.001
        self.min_size = 5

        try:
            f = open('SimpleRNNweight.save', 'r')
            self.iteration_num = pickle.load(f)
            self.emb = pickle.load(f)
            self.weight_x = pickle.load(f)
            self.weight_h = pickle.load(f)
            self.weight_o = pickle.load(f)
            self.bias_h = pickle.load(f)
            self.bias_o = pickle.load(f)
            self.hidden0 = pickle.load(f)
            f.close()
        except:
            self.initialize()

        self.params = [self.emb, self.weight_x, self.weight_h, self.weight_o, self.bias_h, self.bias_o, self.hidden0]

        # paras
        idxs = T.imatrix()      # which word
        x = self.emb[idxs].reshape((idxs.shape[0], self.de*self.cs))
        y = T.iscalar('y')      # label

        def recurrence(x_t, h_tm1):
            h_t = T.nnet.sigmoid(T.dot(x_t, self.weight_x) + T.dot(h_tm1, self.weight_h) + self.bias_h)
            s_t = T.nnet.sigmoid(T.dot(h_t, self.weight_o) + self.bias_o)
            return [h_t, s_t]

        [h, s], _ = theano.scan(fn=recurrence, sequences=x, outputs_info=[self.hidden0, None], n_steps=x.shape[0])

        p_y_given_x_lastword = s[-1]
        y_pred = T.argmax(p_y_given_x_lastword)

        # updates
        lr = T.scalar('r')
        nll = -T.log(p_y_given_x_lastword)[y]
        gradients = T.grad(nll, self.params)
        updates = OrderedDict((p, p-lr*g) for p, g in zip(self.params, gradients))

        # functions
        self.theano_train = theano.function(inputs=[idxs, y, lr], outputs=nll, updates=updates)
        self.theano_normalize = theano.function(inputs=[], updates={self.emb: self.emb/T.sqrt((self.emb**2).sum(axis=1)).dimshuffle(0, 'x')})
        self.theano_classify = theano.function(inputs=[idxs], outputs=y_pred)
        # self.theano_encode = theano.function(inputs=[idxs], outputs=h)

    def initialize(self):
        self.iteration_num = 0
        self.emb = theano.shared(numpy.random.uniform(-1.0, 1.0, (self.ne, self.de)))
        self.weight_x = theano.shared(numpy.random.uniform(-1.0, 1.0, (self.de*self.cs, self.nh)))
        self.weight_h = theano.shared(numpy.random.uniform(-1.0, 1.0, (self.nh, self.nh)))
        self.weight_o = theano.shared(numpy.random.uniform(-1.0, 1.0, (self.nh, self.nc)))
        self.bias_h = theano.shared(numpy.zeros(self.nh, dtype=theano.config.floatX))
        self.bias_o = theano.shared(numpy.zeros(self.nc, dtype=theano.config.floatX))
        self.hidden0 = theano.shared(numpy.zeros(self.nh, dtype=theano.config.floatX))

    def closeEvent(self):
        f = open('SimpleRNNweight.save', 'w')
        pickle.dump(self.iteration_num, f)
        pickle.dump(self.emb, f)
        pickle.dump(self.weight_x, f)
        pickle.dump(self.weight_h, f)
        pickle.dump(self.weight_o, f)
        pickle.dump(self.bias_h, f)
        pickle.dump(self.bias_o, f)
        pickle.dump(self.hidden0, f)
        f.close()

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