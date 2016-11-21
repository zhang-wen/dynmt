"""
bi-directional rnn network for encoder
attention rnn network for decoder
"""

from __future__ import division

import time
import random
import sys
import os

from collections import OrderedDict
import numpy as np

from gru import GRU

from _gdynet import *
print
#from _dynet import *


class Network(object):

    def __init__(
        self,
        swemb_dims,
        twemb_dims,
        enc_hidden_units,
        dec_hidden_units,
        align_dims,
        logistic_in_dims,
        src_vocab_size=30000,
        trg_vocab_size=30000,
        droprate=0.,
    ):

        self.swemb_dims = swemb_dims
        self.twemb_dims = twemb_dims
        self.enc_hidden_units = enc_hidden_units
        self.dec_hidden_units = dec_hidden_units
        self.align_dims = align_dims
        self.logistic_in_dims = logistic_in_dims
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size

        self.droprate = droprate

        self.model = Model()

        self.trainer = AdadeltaTrainer(self.model, eps=1e-6, rho=0.95)
        random.seed(1)

        self.src_lookup_table_name = 'src_emb_lookuptable'
        self.trg_lookup_table_name = 'trg_emb_lookuptable'

        self.dec_1_Wz_name = 'GRU_Dec_Wz_1'
        self.dec_1_Wr_name = 'GRU_Dec_Wr_1'
        self.dec_1_Wh_name = 'GRU_Dec_Wh_1'

        self.attention_W_1_name = 'attention_W_1'
        self.attention_W_2_name = 'attention_W_2'
        self.attention_v_name = 'attention_v'

        self.dec_2_Wz_name = 'GRU_Dec_Wz_2'
        self.dec_2_Wr_name = 'GRU_Dec_Wr_2'
        self.dec_2_Wh_name = 'GRU_Dec_Wh_2'
        self.dec_2_h0_name = 'GRU_Dec_h0_2'

        self.combine_out_W_name = 'combine_out_W'
        self.combine_out_b_name = 'combine_out_b'

        self.logistic_W_name = 'logistic_W'
        self.logistic_b_name = 'logistic_b'

        #self.activation = rectify
        self.activation = tanh

        self.fwd_gru = GRU(self.model, swemb_dims,
                           enc_hidden_units, prefix='GRU_Enc_fwd')
        self.bwd_gru = GRU(self.model, swemb_dims,
                           enc_hidden_units, prefix='GRU_Enc_bwd')

        '''
        self.dec_gru1 = GRU(self.model, twemb_dims,
                            dec_hidden_units, prefix='dec_gru')
        self.dec_gru2 = GRU(self.model, 2 * enc_hidden_units,
                            dec_hidden_units, prefix='dec_gru')
        '''

    def init_params(self):

        self.lp_src_lookup_table = self.model.add_lookup_parameters(
            (self.src_vocab_size, self.swemb_dims)
        )

        self.lp_trg_lookup_table = self.model.add_lookup_parameters(
            (self.trg_vocab_size, self.twemb_dims)
        )

        scale = 0.01

        self.p_dec_1_W_z = self.model.add_parameters(
            (self.dec_hidden_units, self.twemb_dims + self.dec_hidden_units),
            init=UniformInitializer(scale)
        )

        self.p_dec_1_W_r = self.model.add_parameters(
            (self.dec_hidden_units, self.twemb_dims + self.dec_hidden_units),
            init=UniformInitializer(scale)
        )

        self.p_dec_1_W_h = self.model.add_parameters(
            (self.dec_hidden_units, self.twemb_dims + self.dec_hidden_units),
            init=UniformInitializer(scale)
        )

        self.p_attention_W_1 = self.model.add_parameters(
            (self.align_dims, 2 * self.enc_hidden_units),
            init=UniformInitializer(scale)
        )
        self.p_attention_W_2 = self.model.add_parameters(
            (self.align_dims, self.dec_hidden_units),
            init=UniformInitializer(scale)
        )
        self.p_attention_v = self.model.add_parameters(
            (1, self.align_dims),
            init=ConstInitializer(0.)
        )

        self.p_dec_2_W_z = self.model.add_parameters(
            (self.dec_hidden_units, self.dec_hidden_units + 2 * self.enc_hidden_units),
            init=UniformInitializer(scale)
        )

        self.p_dec_2_W_r = self.model.add_parameters(
            (self.dec_hidden_units, self.dec_hidden_units + 2 * self.enc_hidden_units),
            init=UniformInitializer(scale)
        )

        self.p_dec_2_W_h = self.model.add_parameters(
            (self.dec_hidden_units, self.dec_hidden_units + 2 * self.enc_hidden_units),
            init=UniformInitializer(scale)
        )

        self.p_dec_2_h0 = self.model.add_parameters(
            (self.dec_hidden_units, ),
            init=ConstInitializer(0.)
        )

        self.p_combine_out_W = self.model.add_parameters(
            (self.logistic_in_dims, self.twemb_dims +
             self.dec_hidden_units + 2 * self.enc_hidden_units),
            init=UniformInitializer(scale)
        )
        self.p_combine_out_b = self.model.add_parameters(
            (self.logistic_in_dims, ),
            init=ConstInitializer(0.)
        )

        self.p_logistic_W = self.model.add_parameters(
            (self.trg_vocab_size, self.logistic_in_dims),
            init=UniformInitializer(scale)
        )
        self.p_logistic_b = self.model.add_parameters(
            (self.trg_vocab_size, ),
            init=ConstInitializer(0.)
        )

        sys.stderr.write('init network parameters done: \n')
        self.params = OrderedDict({})

        self.params[self.src_lookup_table_name] = self.lp_src_lookup_table
        self.params[self.trg_lookup_table_name] = self.lp_trg_lookup_table
        self.params.update(self.fwd_gru.params)
        self.params.update(self.fwd_gru.params)
        self.params.update(self.bwd_gru.params)

        self.params[self.dec_1_Wz_name] = self.p_dec_1_W_z
        self.params[self.dec_1_Wr_name] = self.p_dec_1_W_r
        self.params[self.dec_1_Wh_name] = self.p_dec_1_W_h

        self.params[self.attention_W_1_name] = self.p_attention_W_1
        self.params[self.attention_W_2_name] = self.p_attention_W_2
        self.params[self.attention_v_name] = self.p_attention_v

        self.params[self.dec_2_Wz_name] = self.p_dec_2_W_z
        self.params[self.dec_2_Wr_name] = self.p_dec_2_W_r
        self.params[self.dec_2_Wh_name] = self.p_dec_2_W_h
        self.params[self.dec_2_h0_name] = self.p_dec_2_h0

        self.params[self.combine_out_W_name] = self.p_combine_out_W
        self.params[self.combine_out_b_name] = self.p_combine_out_b

        self.params[self.logistic_W_name] = self.p_logistic_W
        self.params[self.logistic_b_name] = self.p_logistic_b

    def prepare_params(self):

        self.dec_1_W_z = parameter(self.p_dec_1_W_z)
        self.dec_1_W_r = parameter(self.p_dec_1_W_r)
        self.dec_1_W_h = parameter(self.p_dec_1_W_h)

        self.w1 = parameter(self.p_attention_W_1)
        self.w2 = parameter(self.p_attention_W_2)
        self.v = parameter(self.p_attention_v)

        self.dec_2_W_z = parameter(self.p_dec_2_W_z)
        self.dec_2_W_r = parameter(self.p_dec_2_W_r)
        self.dec_2_W_h = parameter(self.p_dec_2_W_h)
        self.dec_2_h0 = parameter(self.p_dec_2_h0)

        self.combine_out_W = parameter(self.p_combine_out_W)
        self.combine_out_b = parameter(self.p_combine_out_b)

        self.logistic_W = parameter(self.p_logistic_W)
        self.logistic_b = parameter(self.p_logistic_b)

    def save(self, filename):
        '''
        Append architecture hyperparameters to end of PyCNN model file.
        '''
        self.model.save(filename)

        with open(filename, 'a') as f:
            f.write('\n')
            f.write('swemb_dims = {}\n'.format(self.swemb_dims))
            f.write('twemb_dims = {}\n'.format(self.twemb_dims))
            f.write('enc_hidden_units = {}\n'.format(self.enc_hidden_units))
            f.write('dec_hidden_units = {}\n'.format(self.dec_hidden_units))
            f.write('align_dims = {}\n'.format(self.align_dims))
            f.write('logistic_in_dims = {}\n'.format(self.logistic_in_dims))
            f.write('src_vocab_size = {}\n'.format(self.src_vocab_size))
            f.write('trg_vocab_size = {}\n'.format(self.trg_vocab_size))
            f.write('droprate = {}\n'.format(self.droprate))

    def load_model(self, filename):
        '''
        Load model from file created by save() function
        '''
        with open(filename) as f:
            f.readline()
            f.readline()
            swemb_dims = int(f.readline().split()[-1])
            twemb_dims = int(f.readline().split()[-1])
            enc_hidden_units = int(f.readline().split()[-1])
            dec_hidden_units = int(f.readline().split()[-1])
            align_dims = int(f.readline().split()[-1])
            logistic_in_dims = int(f.readline().split()[-1])
            src_vocab_size = int(f.readline().split()[-1])
            trg_vocab_size = int(f.readline().split()[-1])

        network = Network(
            swemb_dims=swemb_dims,
            twemb_dims=twemb_dims,
            enc_hidden_units=enc_hidden_units,
            dec_hidden_units=dec_hidden_units,
            align_dims=align_dims,
            logistic_in_dims=logistic_in_dims,
            src_vocab_size=src_vocab_size,
            trg_vocab_size=trg_vocab_size
        )
        network.model.load(filename)

        return network

    def encode(self, src_wids, src_masks=None):

        fwd_state = self.fwd_gru.initial_state()
        bwd_state = self.bwd_gru.initial_state()

        mb_size = src_wids.shape[0]
        src_len = src_wids.shape[1]

        sentences = [lookup_batch(self.lp_src_lookup_table,
                                  [src_wids[i][j] for i in range(mb_size)]) for j in range(src_len)]
        # src_len * (semb_dim, batch_size), type(_gdynet._lookupBatchExpression)
        #sentence = [lookup(self.lp_src_lookup_table, w) for w in src_wids]

        fwd_out = []
        for vec in sentences:
            fwd_state = fwd_state.add_input(vec)
            fwd_one_word = fwd_state.output()
            fwd_out.append(fwd_one_word)

        if src_masks is not None:
            for i, src_mask in enumerate(src_masks):
                fwd_out[i] = fwd_out[i] * src_mask

        bwd_out = []
        for vec in reversed(sentences):
            bwd_state = bwd_state.add_input(vec)
            bwd_one_word = bwd_state.output()
            bwd_out.append(bwd_one_word)

        if src_masks is not None:
            for i, src_mask in enumerate(reversed(src_masks)):
                bwd_out[i] = bwd_out[i] * src_mask

        enc = [concatenate([f, b])
               for (f, b) in zip(fwd_out, bwd_out[::-1])]

        return enc

    def attention(self, enc_vectors, h_t, src_masks=None):

        attention_weights = []
        w2dec = self.w2 * h_t
        for enc_vec in enc_vectors:
            attention_weight = self.v * tanh(self.w1 * enc_vec + w2dec)
            attention_weights.append(attention_weight)

        if src_masks is not None:
            for i, src_mask in enumerate(src_masks):
                attention_weights[i] = attention_weights[i] * src_mask

        attention_probs = softmax(concatenate(attention_weights))

        out = []
        for i in range(len(enc_vectors)):
            enc_vec = enc_vectors[i]
            attention_prob = pick(attention_probs, i)
            out.append(enc_vec * attention_prob)

        output_vec = esum(out)
        return attention_probs, output_vec

    def first_hidden(self, trg_bwids, s_tm1, trg_mask_expr=None):

        y_tm1_emb = lookup_batch(self.lp_trg_lookup_table, list(trg_bwids))

        x = concatenate([s_tm1, y_tm1_emb])
        z = logistic(self.dec_1_W_z * x)
        r = logistic(self.dec_1_W_r * x)
        _h_t = tanh(self.dec_1_W_h *
                    concatenate(
                        [cwise_multiply(r, s_tm1), y_tm1_emb])
                    )
        h_t = cwise_multiply((1 - z), s_tm1) + cwise_multiply(z, _h_t)

        if trg_mask_expr is not None:
            h_t = h_t * trg_mask_expr + s_tm1 * (1. - trg_mask_expr)

        return y_tm1_emb, h_t

    def next_state(self, attent_vec, h_t, trg_mask_expr=None):

        x = concatenate([h_t, attent_vec])
        z = logistic(self.dec_2_W_z * x)
        r = logistic(self.dec_2_W_r * x)
        _h_t = tanh(self.dec_2_W_h *
                    concatenate(
                        [cwise_multiply(r, h_t), attent_vec])
                    )
        s_t = cwise_multiply((1 - z), h_t) + cwise_multiply(z, _h_t)

        if trg_mask_expr is not None:
            s_t = s_t * trg_mask_expr + h_t * (1. - trg_mask_expr)

        return s_t

    def comb_out(self, y_tm1_emb, s_t, a_t):

        combine = concatenate([y_tm1_emb, s_t, a_t])

        comb_out = self.activation(
            self.combine_out_W * combine + self.combine_out_b)

        return comb_out

    def scores(self, comb_out, part=None):

        if part is not None:
            raise NotImplementedError
        else:
            scores = self.logistic_W * comb_out + self.logistic_b

        return scores

    def softmax(self, scores):

        return softmax(scores)

    def step_with_attention(self, s_tm1, y_tm1_emb, context):

        _, h_t = first_hidden(y_tm1_emb, s_tm1)

        _, attent_vec = self.attention(context, h_t)

        s_t = next_state(attent_vec, h_t)

        combine = concatenate([y_tm1_emb, s_t, attent_vec])

        comb_out = self.activation(
            self.combine_out_W * combine + self.combine_out_b)

    def forward_with_attention(self, trg_mb_len, enc_vectors, src_mask_exprs, trg_mask_exprs):

        output_vec = []

        s_t = self.dec_2_h0

        trg_len_bch = np.transpose(trg_mb_len)

        for trg_bch_wids, mask in zip(trg_len_bch[:-1], trg_mask_exprs[:-1]):

            y_tm1_emb, h_t = self.first_hidden(trg_bch_wids, s_t, mask)

            _, attent_vec = self.attention(enc_vectors, h_t, src_mask_exprs)

            s_t = self.next_state(attent_vec, h_t, mask)

            combine = concatenate([y_tm1_emb, s_t, attent_vec])

            comb_out = self.activation(
                self.combine_out_W * combine + self.combine_out_b)

            output_vec.append(comb_out * mask)

        return output_vec

    def batch_ce(self, input_vectors, golden_targets, trg_mask_exprs):

        output_vec = []
        losses = []
        for (vec, gid, mask) in zip(input_vectors, golden_targets[1:], trg_mask_exprs):
            scores = self.logistic_W * vec + self.logistic_b
            loss = pickneglogsoftmax_batch(scores, gid)
            loss = loss * mask
            losses.append(loss)
        loss = sum_batches(esum(losses))

        return loss

    def get_loss(self, src_mb_len, src_masks, trg_mb_len, trg_masks):

        self.mb_size = len(src_mb_len)

        src_mask_exprs = []
        trg_mask_exprs = []
        for src_mask, trg_mask in zip(np.transpose(src_masks), np.transpose(trg_masks)):
            src_mask_expr = inputVector(src_mask)
            src_mask_exprs.append(reshape(src_mask_expr, (1,), self.mb_size))

            trg_mask_expr = inputVector(trg_mask)
            trg_mask_exprs.append(reshape(trg_mask_expr, (1,), self.mb_size))

        src_enc = self.encode(src_mb_len, src_mask_exprs)
        comb_out = self.forward_with_attention(
            trg_mb_len, src_enc, src_mask_exprs, trg_mask_exprs)
        loss = self.batch_ce(comb_out, np.transpose(
            trg_mb_len), trg_mask_exprs)

        return loss
