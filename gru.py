from collections import OrderedDict

import numpy as np

import logging
logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)
logger.setLevel(logging.INFO)
import os

from _gdynet import *
print
#from _dynet import *


class GRU(object):
    '''
    Gated Recurrent Unit network with initial state as parameter:
        z_t = sigmoid(W_z dot [h_{t-1}, x_t])
        r_t = sigmoid(W_r dot [h_{t-1}, x_t])
        _h_t = tanh(W_h dot [r_t * h_{t-1}, x_t])
        h_t = (1 - z_t) * h_{t-1} + z_t * _h_t
    all parameters are initialized in [-0.01, 0.01]
    '''

    number = 0

    def __init__(self, model, input_dims, output_dims, prefix='GRU'):

        self.input_dims = input_dims
        self.output_dims = output_dims
        self.model = model

        GRU.number += 1
        self.name = '{}_{}'.format(prefix, GRU.number)

        self.W_z_xh_name = '{}_W_z'.format(self.name)

        self.W_r_xh_name = '{}_W_r'.format(self.name)

        self.W_h_xh_name = '{}_W_h'.format(self.name)

        self.h0_name = '{}_h0'.format(self.name)

        scale = 0.01
        # UniformInitializer(scale): uniform between -scale and scale.
        self.p_W_z_xh = self.model.add_parameters(
            (output_dims, output_dims + input_dims),
            init=UniformInitializer(scale)
        )

        self.p_W_r_xh = self.model.add_parameters(
            (output_dims, output_dims + input_dims),
            init=UniformInitializer(scale)
        )

        self.p_W_h_xh = self.model.add_parameters(
            (output_dims, output_dims + input_dims),
            init=UniformInitializer(scale)
        )

        self.p_h0 = self.model.add_parameters(
            (output_dims, ),
            init=ConstInitializer(0.)
        )

        self.params = OrderedDict({})

        self.params[self.W_z_xh_name] = self.p_W_z_xh
        self.params[self.W_r_xh_name] = self.p_W_r_xh
        self.params[self.W_h_xh_name] = self.p_W_h_xh
        self.params[self.h0_name] = self.p_h0

    class State(object):

        def __init__(self, gru):
            self.gru = gru

            self.h_outputs = []

            self.W_z_xh = parameter(self.gru.p_W_z_xh)

            self.W_r_xh = parameter(self.gru.p_W_r_xh)

            self.W_h_xh = parameter(self.gru.p_W_h_xh)

            self.h = parameter(self.gru.p_h0)

            # self.h_outputs.append(self.h)

        def add_input(self, input_vec):

            x = concatenate([self.h, input_vec])

            z = logistic(self.W_z_xh * x)

            r = logistic(self.W_r_xh * x)
            _h = tanh(self.W_h_xh *
                      concatenate(
                          [cwise_multiply(r, self.h), input_vec])
                      )

            h = cwise_multiply((1 - z), self.h) + cwise_multiply(z, _h)

            self.h = h

            self.h_outputs.append(h)

            return self

        def output(self):
            return self.h_outputs[-1]

    def initial_state(self):
        return GRU.State(self)
