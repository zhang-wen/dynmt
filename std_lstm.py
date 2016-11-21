#from _gdynet import *
#from _dynet import *

import pycnn
import numpy as np

import logging
logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)
logger.setLevel(logging.INFO)


class STD_LSTM(object):
    '''
    Normal Long Short Term Memory network with initial state as parameter:
        f_t = sigmoid(W_f dot [h_{t-1}, x_t] + b_f)
        i_t = sigmoid(W_i dot [h_{t-1}, x_t] + b_i)
        _c_t = tanh(W_c dot [h_{t-1}, x_t] + b_c)
        c_t = f_t * c_{t-1} + i_t * _c_t
        o_t = sigmoid(W_o dot [h_{t-1}, x_t] + b_o)
        h_t = o_t * tanh(c_t)
    all parameters are initialized in [-0.01, 0.01]
    '''

    number = 0

    def __init__(self, model, input_dims, output_dims, prefix='STD_LSTM'):
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.model = model

        STD_LSTM.number += 1
        self.name = '{}_{}'.format(prefix, STD_LSTM.number)

        self.W_f_xh_name = '{}_W_f'.format(self.name)
        self.b_f_h_name = '{}_b_f'.format(self.name)

        self.W_i_xh_name = '{}_W_i'.format(self.name)
        self.b_i_h_name = '{}_b_i'.format(self.name)

        self.W_c_xh_name = '{}_W_c'.format(self.name)
        self.b_c_h_name = '{}_b_c'.format(self.name)

        self.W_o_xh_name = '{}_W_o'.format(self.name)
        self.b_o_h_name = '{}_b_o'.format(self.name)

        self.c0_name = '{}_c0'.format(self.name)

        self.model.add_parameters(
            self.W_f_xh_name,
            (output_dims, output_dims + input_dims)
        )

        self.model.add_parameters(
            self.b_f_h_name,
            output_dims
        )

        self.model.add_parameters(
            self.W_i_xh_name,
            (output_dims, output_dims + input_dims)
        )

        self.model.add_parameters(
            self.b_i_h_name,
            output_dims
        )

        self.model.add_parameters(
            self.W_c_xh_name,
            (output_dims, output_dims + input_dims)
        )

        self.model.add_parameters(
            self.b_c_h_name,
            output_dims
        )

        self.model.add_parameters(
            self.W_o_xh_name,
            (output_dims, output_dims + input_dims)
        )

        self.model.add_parameters(
            self.b_o_h_name,
            output_dims
        )

        self.model.add_parameters(
            self.c0_name,
            output_dims
        )

        self.W_f_xh = self.model[self.W_f_xh_name]
        self.b_f_h = self.model[self.b_f_h_name]

        self.W_i_xh = self.model[self.W_i_xh_name]
        self.b_i_h = self.model[self.b_i_h_name]

        self.W_c_xh = self.model[self.W_c_xh_name]
        self.b_c_h = self.model[self.b_c_h_name]

        self.W_o_xh = self.model[self.W_o_xh_name]
        self.b_o_h = self.model[self.b_o_h_name]

        self.c0 = self.model[self.c0_name]

        self.W_f_xh.load_array(np.random.uniform(-0.01, 0.01, self.W_f_xh.shape()))
        self.b_f_h.load_array(np.zeros(self.b_f_h.shape()))

        self.W_i_xh.load_array(np.random.uniform(-0.01, 0.01, self.W_i_xh.shape()))
        self.b_i_h.load_array(np.zeros(self.b_i_h.shape()))

        self.W_c_xh.load_array(np.random.uniform(-0.01, 0.01, self.W_c_xh.shape()))
        self.b_c_h.load_array(np.zeros(self.b_c_h.shape()))

        self.W_o_xh.load_array(np.random.uniform(-0.01, 0.01, self.W_o_xh.shape()))
        self.b_o_h.load_array(np.zeros(self.b_o_h.shape()))

        self.c0.load_array(np.zeros(self.c0.shape()))

    class State(object):

        def __init__(self, std_lstm):
            self.std_lstm = std_lstm

            self.c_outputs = []
            self.h_outputs = []

            self.W_f_xh = pycnn.parameter(self.std_lstm.W_f_xh)
            self.b_f_h = pycnn.parameter(self.std_lstm.b_f_h)

            self.W_i_xh = pycnn.parameter(self.std_lstm.W_i_xh)
            self.b_i_h = pycnn.parameter(self.std_lstm.b_i_h)

            self.W_c_xh = pycnn.parameter(self.std_lstm.W_c_xh)
            self.b_c_h = pycnn.parameter(self.std_lstm.b_c_h)

            self.W_o_xh = pycnn.parameter(self.std_lstm.W_o_xh)
            self.b_o_h = pycnn.parameter(self.std_lstm.b_o_h)

            self.c = pycnn.parameter(self.std_lstm.c0)
            self.h = pycnn.tanh(self.c)

        def add_input(self, input_vec):

            x = pycnn.concatenate([self.h, input_vec])

            f = pycnn.logistic(self.W_f_xh * x + self.b_f_h)
            i = pycnn.logistic(self.W_i_xh * x + self.b_i_h)
            _c = pycnn.tanh(self.W_c_xh * x + self.b_c_h)

            c = pycnn.cwise_multiply(f, c) + pycnn.cwise_multiply(i, _c)

            o = pycnn.logistic(self.W_o_xh * x + self.b_o_h)
            h = pycnn.cwise_multiply(o, pycnn.tanh(c))

            self.c = c
            self.h = h

            self.c_outputs.append(self.c)
            self.h_outputs.append(self.h)

            return self

        def output(self):
            return self.c_outputs[-1], self.h_outputs[-1]

    def initial_state(self):
        return STD_LSTM.State(self)
