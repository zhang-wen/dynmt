#from _gdynet import *
#from _dynet import *

import pycnn
import numpy as np

import logging
logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)
logger.setLevel(logging.INFO)


class STD_RNN(object):
    '''
    Simple Recurrent Neural Network with initial state as parameter:
        s_t = tanh(U dot x_t + W dot s_{t-1} + b)
    all parameters are initialized in [-0.01, 0.01]
    '''

    number = 0

    def __init__(self, model, input_dims, output_dims, prefix='Simple_RNN'):
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.model = model

        STD_RNN.number += 1
        self.name = '{}_{}'.format(prefix, STD_RNN.number)

        self.W_xh_name = '{}_W_xh'.format(self.name)
        self.W_hh_name = '{}_W_hh'.format(self.name)
        self.b_h_name = '{}_b'.format(self.name)

        self.h0_name = '{}_h0'.format(self.name)

        self.model.add_parameters(
            self.W_xh_name,
            (output_dims, input_dims)
        )

        self.model.add_parameters(
            self.W_hh_name,
            (output_dims, output_dims)
        )

        self.model.add_parameters(
            self.b_h_name,
            output_dims
        )

        self.model.add_parameters(
            self.h0_name,
            output_dims
        )

        self.W_xh = self.model[self.W_xh_name]
        self.W_hh = self.model[self.W_hh_name]
        self.b_h = self.model[self.b_h_name]

        self.h0 = self.model[self.h0_name]

        self.W_xh.load_array(np.random.uniform(-0.01, 0.01, self.W_xh.shape()))
        self.W_hh.load_array(np.random.uniform(-0.01, 0.01, self.W_hh.shape()))
        self.b_h.load_array(np.zeros(self.b_h.shape()))

        self.h0.load_array(np.zeros(self.h0.shape()))

    class State(object):

        def __init__(self, std_rnn):
            self.std_rnn = std_rnn

            self.outputs = []

            self.W_xh = pycnn.parameter(self.std_rnn.W_xh)
            self.W_hh = pycnn.parameter(self.std_rnn.W_hh)
            self.b_h = pycnn.parameter(self.std_rnn.b_h)

            self.h = self.std_rnn.h0

        def add_input(self, input_vec):

            h = pycnn.tanh(self.W_xh * input_vec + self.W_hh * self.h + self.b_h)
            self.h = h

            self.outputs.append(h)

            return self

        def output(self):
            return self.outputs[-1]

    def initial_state(self):
        return STD_RNN.State(self)


class ML_STD_RNN(object):

    def __init__(self, model, input_dims, output_dims, num_of_layers=1):
        self.layers = []
        for i in range(num_of_layers):
            l_rnn = STD_RNN(model, input_dims, output_dims, prefix='Simple_RNN_L')
            r_rnn = STD_RNN(model, input_dims, output_dims, prefix='Simple_RNN_R')
