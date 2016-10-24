__author__ = 'liushuman'

import theano
import theano.tensor as T

import sys
sys.path.append("..")
from utils.ActivationFunctions import ActivationFunctions
from utils.Params import Params


class GRUUnit(object):

    def __init__(self, wx_size, wh_size, h_size, bias_size=None, context=None):
        # context is an encode result(matrix) in h_size to add to the gru
        self.reset_gate = ResetGate(wx_size, wh_size, bias_size, context)
        self.update_gate = UpdateGate(wx_size, wh_size, bias_size, context)

        self.weight_xt_candidate = Params().uniform(wx_size)
        self.weight_htm1_candidate = Params().uniform(wh_size)
        self.hidden0 = Params().constant(h_size)

        self.params = self.reset_gate.params + self.update_gate.params + \
            [self.weight_xt_candidate, self.weight_htm1_candidate, self.hidden0]

        if bias_size is not None:
            self._if_bias = True
            self.bias = Params().constant(bias_size)
            self.params.append(self.bias)
        else:
            self._if_bias = False

        if context is not None:
            self._if_context = True
            self.weight_context = Params().uniform(wh_size)
            self.context = T.dot(Params().new(context), self.weight_context)
            # self.context = T.dot(self.weight_context, Params().new(context)).reshape((1, wh_size[1]))
            self.params.append(self.weight_context)
        else:
            self._if_context = False
            self.context = Params().constant((1, wh_size[1]))

    def apply(self, x_sequences, activate_type='tanh'):
        # function for auto encoder, x_sequences is a src sequence with vocab rows and d_emb cols

        def _step_forward(x_t, h_tm1):
            # x_t is vector, h_tm1 is matrix
            reset_t = self.reset_gate.apply(x_t, h_tm1[0])      # matrix
            update_t = self.update_gate.apply(x_t, h_tm1[0])    # matrix

            candidate_ht = T.dot(self.weight_xt_candidate, x_t) + \
                T.dot(self.weight_htm1_candidate, (reset_t * h_tm1[0])[0]) + self.context

            if self._if_bias:
                candidate_ht += self.bias

            candidate_ht = ActivationFunctions().apply(candidate_ht, activate_type)     # matrix

            ht = (1-update_t) * h_tm1[0] + update_t * candidate_ht[0]   # matrix

            return ht

        hidden_layers, _ = theano.scan(fn=_step_forward, sequences=x_sequences, outputs_info=[self.hidden0])

        return hidden_layers    # a matrix n_step * 1 * cols

    def merge_out(self, x_sequences, wm_size, activate_type='tanh'):
        # function for prediction and auto decoder, x_sequences is the target sequence been generated

        # hidden is a matrix, 1 * n_hid
        # x_sequences is a matrix, x_sequences_size * n_emb
        # context is a matrix, with 1 * n_hid
        hidden = self.apply(x_sequences, activate_type)[-1]

        if self._if_context:
            combine = T.concatenate([hidden[0], x_sequences[-1], self.context[0]])      # vector
        else:
            combine = T.concatenate([hidden[0], x_sequences[-1]])                       # vector

        self.weight_merge_out = Params().uniform(wm_size)
        self.params += self.weight_merge_out

        if self._if_bias:
            self.bias_merge_out = Params().uniform((1, wm_size[0]))
            self.params += self.bias_merge_out

        merge_out = theano.dot(self.weight_merge_out, combine)
        if self._if_bias:
            merge_out += self.bias_merge_out                    # matrix

        # result without classification
        return merge_out


class ResetGate(object):

    def __init__(self, wx_size, wh_size, bias_size=None, context=None):
        self.weight_xt = Params().uniform(wx_size)
        self.weight_htm1 = Params().uniform(wh_size)
        self.params = [self.weight_xt, self.weight_htm1]

        if bias_size is not None:
            self._if_bias = True
            self.bias = Params().constant(bias_size)
            self.params.append(self.bias)
        else:
            self._if_bias = False

        if context is not None:
            self.weight_context = Params().uniform(wh_size)
            self.context = T.dot(Params().new(context), self.weight_context)  # .reshape((1, wh_size[1]))
            self.params.append(self.weight_context)
        else:
            self.context = Params().constant((1, wh_size[1]))

    def apply(self, xt, htm1, activate_type='sigmoid'):
        gate_output = T.dot(self.weight_xt, xt) + T.dot(self.weight_htm1, htm1) + self.context
        if self._if_bias:
            gate_output += self.bias

        gate_output = ActivationFunctions().apply(gate_output, activate_type)

        return gate_output      # matrix


class UpdateGate(object):

    def __init__(self, wx_size, wh_size, bias_size=None, context=None):
        self.weight_xt = Params().uniform(wx_size)
        self.weight_htm1 = Params().uniform(wh_size)
        self.params = [self.weight_xt, self.weight_htm1]

        if bias_size is not None:
            self._if_bias = True
            self.bias = Params().constant(bias_size)
            self.params.append(self.bias)
        else:
            self._if_bias = False

        if context is not None:
            self.weight_context = Params().uniform(wh_size)
            self.context = T.dot(Params().new(context), self.weight_context)  # .reshape((1, wh_size[1]))
            self.params.append(self.weight_context)
        else:
            self.context = Params().constant((1, wh_size[1]))

    def apply(self, xt, htm1, activate_type='sigmoid'):
        gate_output = T.dot(self.weight_xt, xt) + T.dot(self.weight_htm1, htm1) + self.context
        if self._if_bias:
            gate_output += self.bias

        gate_output = ActivationFunctions().apply(gate_output, activate_type)

        return gate_output      # matrix