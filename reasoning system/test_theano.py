__author__ = 'liushuman'

import theano.tensor as T

from collections import OrderedDict
from numpy import *
from theano import *

cs = 3             # word window context size      4 RNN steps & 4*20 di vector
ne = 3             # number of emb & input
nh = 2             # number of hidden
de = 2             # dimension of emb
nc = 3             # number of class & output
learning_rate = 0.01
window_size = 0

emb = theano.shared(numpy.random.uniform(-1.0, 1.0, (ne, de)))
#weight_x = theano.shared(numpy.random.uniform(-1.0, 1.0, (de*cs, nh)))
weight_x = theano.shared(array([[1, 1],
                          [1, 1],
                          [1, 1],
                          [1, 1],
                          [1, 1],
                          [1, 1]]).astype(theano.config.floatX))
#weight_h = theano.shared(numpy.random.uniform(-1.0, 1.0, (nh, nh)))
weight_h = theano.shared(array([[1, 1],
                               [1, 1]]).astype(theano.config.floatX))
#weight_o = theano.shared(numpy.random.uniform(-1.0, 1.0, (nh, nc)))
weight_o = theano.shared(array([[1, 1, 1],
                               [1, 1, 1]]).astype(theano.config.floatX))
bias_h = theano.shared(numpy.zeros(nh, dtype=theano.config.floatX))
bias_o = theano.shared(numpy.zeros(nc, dtype=theano.config.floatX))
hidden0 = theano.shared(numpy.zeros(nh, dtype=theano.config.floatX))

idxs = T.imatrix()
x = emb[idxs].reshape((idxs.shape[0], de*cs))

def recurrence(x_t, h_tm1):
    h_t = T.nnet.sigmoid(T.dot(x_t, weight_x) + T.dot(h_tm1, weight_h) + bias_h)
    s_t = T.nnet.sigmoid(T.dot(h_t, weight_o) + bias_o)
    return [h_t, s_t]

[h, s], _ = theano.scan(fn=recurrence, sequences=x, outputs_info=[hidden0, None], n_steps=x.shape[0])


test_print_emb = theano.function(inputs=[], outputs=emb)
test_print_weight_x = theano.function(inputs=[], outputs=weight_x)
test_print_weight_h = theano.function(inputs=[], outputs=weight_h)
test_print_weight_o = theano.function(inputs=[], outputs=weight_o)
test_print_bias_h = theano.function(inputs=[], outputs=bias_h)
test_print_bias_o = theano.function(inputs=[], outputs=bias_o)
test_print_hidden0 = theano.function(inputs=[], outputs=hidden0)


test_idxs = theano.function(inputs=[idxs], outputs=idxs)
test_x = theano.function(inputs=[idxs], outputs=x)
test_s = theano.function(inputs=[idxs], outputs=s)
test_s2 = theano.function(inputs=[idxs], outputs=s[-1])


print "================================"
print "================emb================"
print test_print_emb()
print "================================"
print "================weight_x================"
print test_print_weight_x()
print "================================"
print "================weight_h================"
print test_print_weight_h()
print "================================"
print "================weight_o================"
print test_print_weight_o()
print "================================"
print "================bias_h================"
print test_print_bias_h()
print "================================"
print "================bias_o================"
print test_print_bias_o()
print "================================"
print "================hidden0================"
print test_print_hidden0()

print ' '
print ' '
print ' '

input = [[0, 1, 2],
         [1, 2, 0]]
print "================================"
print "================idxs================"
print test_idxs(input)
print "================================"
print "================x================"
print test_x(input)
print "================================"
print "================s================"
print test_s(input)
print test_s2(input)