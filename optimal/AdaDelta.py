__author__ = 'liushuman'

import itertools
import numpy
import theano
import theano.tensor as T


class AdaDelta(object):

    def __init__(self, params, rho=0.95, eps=1e-6):
        self._gradients_accumulate = [theano.shared(numpy.zeros(p.get_value().shape,
                                                                dtype=theano.config.floatX)) for p in params]
        self._delta_accumulate = [theano.shared(numpy.zeros(p.get_value().shape,
                                                            dtype=theano.config.floatX)) for p in params]
        self._rho = rho
        self._eps = eps
        self.params = [self._gradients_accumulate, self._delta_accumulate]

    def apply(self, params, gradients):
        gradients_updates = [(_g_a, self._rho * _g_a + (1 - self._rho) * (_g**2))
                             for _g_a, _g in itertools.izip(self._gradients_accumulate, gradients)]

        deltas = [(T.sqrt(_d_a + self._eps)/T.sqrt(_g_a + self._eps)) * _grad for _d_a, _g_a, _grad
                  in itertools.izip(self._delta_accumulate, self._gradients_accumulate, gradients)]

        delta_updates = [(_d_a, self._rho * _d_a + (1 - self._rho) * (_d**2))
                         for _d_a, _d in itertools.izip(self._delta_accumulate, deltas)]
        params_updates = [(_p, _p + _d) for _p, _d in itertools.izip(params, deltas)]

        updates = gradients_updates + delta_updates + params_updates
        return updates