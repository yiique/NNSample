__author__ = 'liushuman'


import theano
import theano.tensor as T

import sys
sys.path.append("..")
from utils.Params import Params


class Embedding(object):

    def __init__(self, emb_size):
        self.emb = Params().uniform(emb_size)
        self.params = [self.emb]

    def apply(self, indexs):
        output = self.emb[indexs].reshape((indexs.shape[0], indexs.shape[1] * self.emb.shape[1]))
        return output