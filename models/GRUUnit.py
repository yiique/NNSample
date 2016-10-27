__author__ = 'liushuman'

import theano
import theano.tensor as T

import sys
sys.path.append("..")
from utils.ActivationFunctions import ActivationFunctions
from utils.Params import Params


class Gate(object):

    def __init__(self, input_size, hidden_size, if_bias=True, context=None, activate_type='sigmoid'):
        if input_size[1] != hidden_size[1]:
            print "WARNING IN GATE: YOUR INPUT COLUMNS IS NOT EQUAL WITH HIDDEN COLUMNS!"

        self._if_bias = if_bias
        self._context = context
        self._activate_type = activate_type

        self.weight_xt = Params().uniform((hidden_size[0], input_size[0]))
        self.weight_htm1 = Params().uniform((hidden_size[0], hidden_size[0]))
        self.params = [self.weight_xt, self.weight_htm1]
        if self._if_bias:
            self.bias = Params().constant(hidden_size)
            self.params += [self.bias]
        if self._context is not None:
            self.weight_context = Params().uniform((hidden_size[0], hidden_size[0]))
            self._context = T.dot(self.weight_context, self._context)
            self.params += [self.weight_context]

    def apply(self, xt, htm1):
        gate_output = T.dot(self.weight_xt, xt) + T.dot(self.weight_htm1, htm1)
        if self._if_bias:
            gate_output += self.bias
        if self._context is not None:
            gate_output += self._context

        gate_output = ActivationFunctions().apply(gate_output, self._activate_type)

        return gate_output


class GRUUnit(object):

    def __init__(self, input_size, hidden_size, if_bias=True, context=None, activate_type='tanh'):
        if input_size[1] != hidden_size[1]:
            print "WARNING IN GRUUNIT: YOUR INPUT COLUMNS IS NOT EQUAL WITH HIDDEN COLUMNS!"

        self._if_bias = if_bias
        self._context = context
        self._activate_type = activate_type

        self.reset_gate = Gate(input_size, hidden_size, if_bias, context)
        self.update_gate = Gate(input_size, hidden_size, if_bias, context)
        self.weight_xt = Params().uniform((hidden_size[0], input_size[0]))
        self.weight_htm1 = Params().uniform((hidden_size[0], hidden_size[0]))
        self.hidden0 = Params().constant(hidden_size)
        self.params = self.reset_gate.params + self.update_gate.params + \
            [self.weight_xt, self.weight_htm1, self.hidden0]

        if self._if_bias:
            self.bias = Params().constant(hidden_size)
            self.params += [self.bias]
        if self._context is not None:
            self.weight_context = Params().uniform((hidden_size[0], hidden_size[0]))
            self._context = T.dot(self.weight_context, self._context)
            self.params += [self.weight_context]

    def apply(self, sequences):

        def _step_forward(x_t, h_tm1):
            # x_t is a vector, h_tm1 is a matrix
            reset_t = self.reset_gate.apply(x_t, h_tm1)
            update_t = self.update_gate.apply(x_t, h_tm1)
            candidate_ht = T.dot(self.weight_xt, x_t) + T.dot(self.weight_htm1, (reset_t * h_tm1))

            if self._if_bias:
                candidate_ht += self.bias
            if self._context is not None:
                candidate_ht += self._context

            candidate_ht = ActivationFunctions().apply(candidate_ht, self._activate_type)

            ht = (1 - update_t) * h_tm1 + update_t * candidate_ht

            return ht           # matrix in hidden size

        hidden_layers, _ = theano.scan(fn=_step_forward, sequences=sequences, outputs_info=[self.hidden0])
        return hidden_layers    # a tensor3 in n_step * hidden rows * hidden cols

    def merge_out(self, sequences, w_mo_size, merge_out_size):

        hidden = self.apply(sequences)      # tensor3

        if self._context is not None:
            context = T.alloc(self._context, T.shape(hidden)[0], T.shape(self._context)[0], T.shape(self._context)[1])
            combine = T.concatenate([hidden, sequences, context], axis=1)       # tensor3
        else:
            combine = T.concatenate([hidden, sequences], axis=1)                # tensor3
        # return combine

        self.weight_merge_out = Params().uniform(w_mo_size)
        self.params += [self.weight_merge_out]
        if self._if_bias:
            self.bias_merge_out = Params().constant(merge_out_size)
            self.params += [self.bias_merge_out]

        merge_out = theano.dot(self.weight_merge_out, combine)
        # return merge_out
        if self._if_bias:
            merge_out += T.alloc(self.bias_merge_out, T.shape(hidden)[0], T.shape(self.bias_merge_out)[0], T.shape(self.bias_merge_out)[1])

        return merge_out        # a tensor3 in n_step * out rows * out cols

