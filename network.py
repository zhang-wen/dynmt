"""
bi-directional rnn network for encoder
attention rnn network for decoder
"""

from __future__ import division

import time
import random
import sys
import os

import subprocess
import collections
import cPickle as pickle

import numpy as np

import configurations
from utils import init_dirs
from cp_sample import trans_samples
from stream_with_dict import get_tr_stream, ensure_special_tokens
from gru import GRU

from _gdynet import *
print
# from _dynet import *


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
        uniform_params=False,
        clipping_threshold=1.0,
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
        self.uniform_params = uniform_params
        self.clipping_threshold = clipping_threshold

        self.model = Model()

        self.trainer = AdadeltaTrainer(self.model, eps=1e-6, rho=0.95)
        self.trainer.set_clip_threshold(self.clipping_threshold)

        random.seed(1)

        self.src_lookup_table_name = 'src_emb_lookuptable'
        self.trg_lookup_table_name = 'trg_emb_lookuptable'

        self.dec_Wz_name = 'GRU_Dec_W_z'
        self.dec_Uz_name = 'GRU_Dec_U_z'
        self.dec_Cz_name = 'GRU_Dec_C_z'
        self.dec_bz_name = 'GRU_Dec_b_z'

        self.dec_Wr_name = 'GRU_Dec_W_r'
        self.dec_Ur_name = 'GRU_Dec_U_r'
        self.dec_Cr_name = 'GRU_Dec_C_r'
        self.dec_br_name = 'GRU_Dec_b_r'

        self.dec_Wh_name = 'GRU_Dec_W_h'
        self.dec_Uh_name = 'GRU_Dec_U_h'
        self.dec_Ch_name = 'GRU_Dec_C_h'
        self.dec_bh_name = 'GRU_Dec_b_h'

        self.dec_init_Ws_name = 'GRU_Dec_init_Ws'
        self.dec_init_bs_name = 'GRU_Dec_init_bs'

        self.attention_W_name = 'attention_Wa'
        self.attention_U_name = 'attention_Ua'
        self.attention_v_name = 'attention_va'

        self.combine_Uo_name = 'combine_U_o'
        self.combine_Vo_name = 'combine_V_o'
        self.combine_Co_name = 'combine_C_o'
        self.combine_bo_name = 'combine_b_o'

        self.logistic_W0_name = 'logistic_W0'
        self.logistic_b0_name = 'logistic_b0'

        #self.activation = rectify
        self.activation = tanh

        self.scale = 0.01
        uniform_initer = UniformInitializer(self.scale)

        # normal with mean and variance.
        gaussian_initer_e3 = NormalInitializer(mean=0, var=0.001)
        gaussian_initer_e2 = NormalInitializer(mean=0, var=0.01)

        self.W_initer = uniform_initer if uniform_params else gaussian_initer_e2
        self.Wa_initer = uniform_initer if uniform_params else gaussian_initer_e3
        self.b_initer = ConstInitializer(0.)

        self.fwd_gru = GRU(self.model, swemb_dims,
                           enc_hidden_units, self.W_initer, self.b_initer, prefix='GRU_Enc_fwd')
        self.bwd_gru = GRU(self.model, swemb_dims,
                           enc_hidden_units, self.W_initer, self.b_initer, prefix='GRU_Enc_bwd')

    def init_params(self):

        lp_src_shape = (self.src_vocab_size, self.swemb_dims)
        self.lp_src_lookup_table = self.model.add_lookup_parameters(lp_src_shape)

        lp_trg_shape = (self.trg_vocab_size, self.twemb_dims)
        self.lp_trg_lookup_table = self.model.add_lookup_parameters(lp_trg_shape)

        if self.uniform_params:
            self.lp_src_lookup_table.init_from_array(
                np.random.uniform(-self.scale, self.scale, lp_src_shape))
            self.lp_trg_lookup_table.init_from_array(
                np.random.uniform(-self.scale, self.scale, lp_trg_shape))
        else:
            self.lp_src_lookup_table.init_from_array(
                np.random.normal(0., 0.01, lp_src_shape)
            )
            self.lp_trg_lookup_table.init_from_array(
                np.random.normal(0., 0.01, lp_trg_shape)
            )

        self.p_dec_W_z = self.model.add_parameters(
            (self.dec_hidden_units, self.twemb_dims),
            init=self.W_initer
        )
        self.p_dec_U_z = self.model.add_parameters(
            (self.dec_hidden_units, self.dec_hidden_units),
            init=self.W_initer
        )
        self.p_dec_C_z = self.model.add_parameters(
            (self.dec_hidden_units, 2 * self.enc_hidden_units),
            init=self.W_initer
        )
        self.p_dec_b_z = self.model.add_parameters(
            (self.dec_hidden_units, ),
            init=self.b_initer
        )

        self.p_dec_W_r = self.model.add_parameters(
            (self.dec_hidden_units, self.twemb_dims),
            init=self.W_initer
        )
        self.p_dec_U_r = self.model.add_parameters(
            (self.dec_hidden_units, self.dec_hidden_units),
            init=self.W_initer
        )
        self.p_dec_C_r = self.model.add_parameters(
            (self.dec_hidden_units, 2 * self.enc_hidden_units),
            init=self.W_initer
        )
        self.p_dec_b_r = self.model.add_parameters(
            (self.dec_hidden_units, ),
            init=self.b_initer
        )

        self.p_dec_W_h = self.model.add_parameters(
            (self.dec_hidden_units, self.twemb_dims),
            init=self.W_initer
        )
        self.p_dec_U_h = self.model.add_parameters(
            (self.dec_hidden_units, self.dec_hidden_units),
            init=self.W_initer
        )
        self.p_dec_C_h = self.model.add_parameters(
            (self.dec_hidden_units, 2 * self.enc_hidden_units),
            init=self.W_initer
        )
        self.p_dec_b_h = self.model.add_parameters(
            (self.dec_hidden_units, ),
            init=self.b_initer
        )

        self.p_dec_init_W_s = self.model.add_parameters(
            (self.dec_hidden_units, self.dec_hidden_units),
            init=self.W_initer
        )
        self.p_dec_init_b_s = self.model.add_parameters(
            (self.dec_hidden_units, ),
            init=self.b_initer
        )

        self.p_attention_W = self.model.add_parameters(
            (self.align_dims, self.dec_hidden_units),
            init=self.Wa_initer
        )
        self.p_attention_U = self.model.add_parameters(
            (self.align_dims, 2 * self.enc_hidden_units),
            init=self.Wa_initer
        )
        self.p_attention_v = self.model.add_parameters(
            (1, self.align_dims),
            init=self.b_initer
        )

        self.p_comb_U_o = self.model.add_parameters(
            (2 * self.logistic_in_dims, self.dec_hidden_units),
            init=self.W_initer
        )
        self.p_comb_V_o = self.model.add_parameters(
            (2 * self.logistic_in_dims, self.twemb_dims),
            init=self.W_initer
        )
        self.p_comb_C_o = self.model.add_parameters(
            (2 * self.logistic_in_dims, 2 * self.enc_hidden_units),
            init=self.W_initer
        )
        self.p_comb_b_o = self.model.add_parameters(
            (2 * self.logistic_in_dims, ),
            init=self.b_initer
        )

        self.p_logistic_W_0 = self.model.add_parameters(
            (self.trg_vocab_size, 2 * self.logistic_in_dims),
            init=self.W_initer
        )
        self.p_logistic_b_0 = self.model.add_parameters(
            (self.trg_vocab_size, ),
            init=self.b_initer
        )

        sys.stderr.write('init network parameters done: \n')
        self.params = collections.OrderedDict({})

        self.params[self.src_lookup_table_name] = self.lp_src_lookup_table
        self.params[self.trg_lookup_table_name] = self.lp_trg_lookup_table
        self.params.update(self.fwd_gru.params)
        self.params.update(self.bwd_gru.params)

        self.params[self.dec_Wz_name] = self.p_dec_W_z
        self.params[self.dec_Uz_name] = self.p_dec_U_z
        self.params[self.dec_Cz_name] = self.p_dec_C_z
        self.params[self.dec_bz_name] = self.p_dec_b_z

        self.params[self.dec_Wr_name] = self.p_dec_W_r
        self.params[self.dec_Ur_name] = self.p_dec_U_r
        self.params[self.dec_Cr_name] = self.p_dec_C_r
        self.params[self.dec_br_name] = self.p_dec_b_r

        self.params[self.dec_Wh_name] = self.p_dec_W_h
        self.params[self.dec_Uh_name] = self.p_dec_U_h
        self.params[self.dec_Ch_name] = self.p_dec_C_h
        self.params[self.dec_bh_name] = self.p_dec_b_h

        self.params[self.dec_init_Ws_name] = self.p_dec_init_W_s
        self.params[self.dec_init_bs_name] = self.p_dec_init_b_s

        self.params[self.attention_W_name] = self.p_attention_W
        self.params[self.attention_U_name] = self.p_attention_U
        self.params[self.attention_v_name] = self.p_attention_v

        self.params[self.combine_Uo_name] = self.p_comb_U_o
        self.params[self.combine_Vo_name] = self.p_comb_V_o
        self.params[self.combine_Co_name] = self.p_comb_C_o
        self.params[self.combine_bo_name] = self.p_comb_b_o

        self.params[self.logistic_W0_name] = self.p_logistic_W_0
        self.params[self.logistic_b0_name] = self.p_logistic_b_0

    def prepare_params(self):

        self.Wz = parameter(self.p_dec_W_z)
        self.Uz = parameter(self.p_dec_U_z)
        self.Cz = parameter(self.p_dec_C_z)
        self.bz = parameter(self.p_dec_b_z)

        self.Wr = parameter(self.p_dec_W_r)
        self.Ur = parameter(self.p_dec_U_r)
        self.Cr = parameter(self.p_dec_C_r)
        self.br = parameter(self.p_dec_b_r)

        self.Wh = parameter(self.p_dec_W_h)
        self.Uh = parameter(self.p_dec_U_h)
        self.Ch = parameter(self.p_dec_C_h)
        self.bh = parameter(self.p_dec_b_h)

        self.Ws_init = parameter(self.p_dec_init_W_s)
        self.bs_init = parameter(self.p_dec_init_b_s)

        self.Wa = parameter(self.p_attention_W)
        self.Ua = parameter(self.p_attention_U)
        self.va = parameter(self.p_attention_v)

        self.Uo = parameter(self.p_comb_U_o)
        self.Vo = parameter(self.p_comb_V_o)
        self.Co = parameter(self.p_comb_C_o)
        self.bo = parameter(self.p_comb_b_o)

        self.W0 = parameter(self.p_logistic_W_0)
        self.b0 = parameter(self.p_logistic_b_0)

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
            f.write('uniform_params = {}\n'.format(self.uniform_params))
            f.write('clipping_threshold = {}\n'.format(self.clipping_threshold))

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
            uniform_params = bool(f.readline().split()[-1])
            clipping_threshold = float(f.readline().split()[-1])

        network = Network(
            swemb_dims=swemb_dims,
            twemb_dims=twemb_dims,
            enc_hidden_units=enc_hidden_units,
            dec_hidden_units=dec_hidden_units,
            align_dims=align_dims,
            logistic_in_dims=logistic_in_dims,
            src_vocab_size=src_vocab_size,
            trg_vocab_size=trg_vocab_size,
            uniform_params=uniform_params,
            clipping_threshold=clipping_threshold
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

        self.dec_s0 = self.Ws_init * bwd_out[0] + self.bs_init

        enc = [concatenate([f, b])
               for (f, b) in zip(fwd_out, bwd_out[::-1])]

        return enc

    def attention(self, enc_hs, ua_hjs, s_t, src_masks=None):

        attention_weights = []
        wa_s_t = self.Wa * s_t
        for ua_hj in ua_hjs:
            attention_weight = self.va * self.activation(wa_s_t + ua_hj)
            attention_weights.append(attention_weight)

        if src_masks is not None:
            for i, src_mask in enumerate(src_masks):
                attention_weights[i] = attention_weights[i] * src_mask

        attention_probs = softmax(concatenate(attention_weights))

        out = []
        for i in range(len(enc_hs)):
            h_j = enc_hs[i]
            attention_prob = pick(attention_probs, i)
            out.append(h_j * attention_prob)

        output_vec = esum(out)
        return attention_probs, output_vec

    def next_state(self, y_tm1_bwids, c_t, s_tm1, trg_mask_expr=None):

        y_tm1_emb = lookup_batch(self.lp_trg_lookup_table, list(y_tm1_bwids))

        z = logistic(self.Wz * y_tm1_emb + self.Uz * s_tm1 + self.Cz * c_t + self.bz)
        r = logistic(self.Wr * y_tm1_emb + self.Ur * s_tm1 + self.Cr * c_t + self.br)
        _s_t = self.activation(self.Wh * y_tm1_emb +
                               self.Uh * cmult(r, s_tm1) +
                               self.Ch * c_t + self.bh
                               )

        s_t = cmult((1 - z), s_tm1) + cmult(z, _s_t)

        if trg_mask_expr is not None:
            s_t = s_t * trg_mask_expr + s_tm1 * (1. - trg_mask_expr)

        return y_tm1_emb, s_t

    def comb_out(self, s_t, y_tm1_emb, c_t, mask=None):

        _t_i = self.Uo * s_t + self.Vo * y_tm1_emb + self.Co * c_t + self.bo

        if mask is not None:
            _t_i = _t_i * mask

        return _t_i

    def next_scores(self, comb_out, part=None):

        if part is not None:
            raise NotImplementedError
        else:
            scores = self.W0 * comb_out + self.b0

        return scores

    def cross_entropy(self, scores):

        return -log_softmax(scores)

    def init_ua_hjs(self, enc_hs):

        ua_hjs = []
        for enc_hj in enc_hs:
            ua_hjs.append(self.Ua * enc_hj)

        return ua_hjs

    def forward_with_attention(self, trg_mb_len, enc_hs, src_mask_exprs, trg_mask_exprs):

        output_vec = []

        s_i = self.dec_s0
        ua_hjs = self.init_ua_hjs(enc_hs)

        trg_len_bch = np.transpose(trg_mb_len)

        for trg_bch_wids, mask in zip(trg_len_bch[:-1], trg_mask_exprs[:-1]):

            _, attent_vec = self.attention(enc_hs, ua_hjs, s_i, src_mask_exprs)

            y_im1_emb, s_i = self.next_state(trg_bch_wids, attent_vec, s_i, mask)

            t_i = self.comb_out(s_i, y_im1_emb, attent_vec, mask)

            output_vec.append(t_i)

        return output_vec

    def batch_ce(self, input_vectors, golden_targets, trg_mask_exprs):

        output_vec = []
        losses = []
        for (vec, gid, mask) in zip(input_vectors, golden_targets[1:], trg_mask_exprs):
            scores = self.next_scores(vec)
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

    @staticmethod
    def train():

        start_time = time.time()

        config = getattr(configurations, 'get_config_cs2en')()

        from manvocab import topk_target_vcab_list
        ltopk_trg_vocab_idx = topk_target_vcab_list(**config)

        sys.stderr.write('\nload source and target vocabulary ...\n')
        sys.stderr.write('want to generate source dict {} and target dict {}: \n'.format(
            config['src_vocab_size'], config['trg_vocab_size']))
        src_vocab = pickle.load(open(config['src_vocab']))
        trg_vocab = pickle.load(open(config['trg_vocab']))
        # for k, v in src_vocab.iteritems():
        #    print k, v
        sys.stderr.write('~done source vocab count: {}, target vocab count: {}\n'.format(
            len(src_vocab), len(trg_vocab)))
        sys.stderr.write('vocabulary contains <S>, <UNK> and </S>\n')

        seos_idx, teos_idx = config['src_vocab_size'] - \
            1, config['trg_vocab_size'] - 1
        src_vocab = ensure_special_tokens(
            src_vocab, bos_idx=0, eos_idx=seos_idx, unk_idx=config['unk_id'])
        trg_vocab = ensure_special_tokens(
            trg_vocab, bos_idx=0, eos_idx=teos_idx, unk_idx=config['unk_id'])

        # the trg_vocab is originally:
        #   {'UNK': 1, '<s>': 0, '</s>': 0, 'is': 5, ...}
        # after ensure_special_tokens, the trg_vocab becomes:
        #   {'<UNK>': 1, '<S>': 0, '</S>': trg_vocab_size-1, 'is': 5, ...}
        trg_vocab_i2w = {index: word for word, index in trg_vocab.iteritems()}
        src_vocab_i2w = {index: word for word, index in src_vocab.iteritems()}
        # after reversing, the trg_vocab_i2w become:
        #   {1: '<UNK>', 0: '<S>', trg_vocab_size-1: '</S>', 5: 'is', ...}

        init_dirs(**config)

        sys.stderr.write('done\n')

        nwk = Network(
            swemb_dims=config['swemb_dims'],
            twemb_dims=config['twemb_dims'],
            enc_hidden_units=config['enc_hidden_units'],
            dec_hidden_units=config['dec_hidden_units'],
            align_dims=config['align_dims'],
            logistic_in_dims=config['logistic_in_dims'],
            uniform_params=config['uniform_params'],
            clipping_threshold=config['clipping_threshold'],
        )
        nwk.init_params()

        for name, param in nwk.params.iteritems():
            sys.stderr.write('    {:20}: {}\n'.format(name, param.as_array().shape))

        k_batch_start_sample = config['k_batch_start_sample']
        batch_size, sample_size = config['batch_size'], config['hook_samples']
        if batch_size < sample_size:
            sys.stderr.write(
                'batch size must be great or equal with sample size')
            sys.exit(0)

        batch_start_sample = np.random.randint(
            2, k_batch_start_sample)  # [low, high)
        sys.stderr.write('will randomly generate {} sample at {}th batch\n'.format(
            sample_size, batch_start_sample))

        batch_count, sent_count, val_time, best_score = 0, 0, 0, 0.
        model_name = ''
        sample_src_np, sample_trg_np = None, None

        sample_switch = [0, config['use_batch'], config[
            'use_score'], config['use_norm'], config['use_mv'], config['merge_way']]
        beam_size = config['beam_size']
        search_mode = config['search_mode']

        start_time = time.time()
        tr_stream = get_tr_stream(**config)
        sys.stderr.write('Start training!!!\n')
        max_epochs = config['max_epoch']

        batch_sub_vocab = None
        batch_vcb_fix = None

        for epoch in range(max_epochs):
            # take the batch sizes 3 as an example:
            # tuple: tuple[0] is indexes of source sentence (np.ndarray)
            # like array([[0, 23, 3, 4, 29999], [0, 2, 1, 29999], [0, 31, 333, 2, 1, 29999]])
            # tuple: tuple[1] is indexes of source sentence mask (np.ndarray)
            # like array([[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]])
            # tuple: tuple[2] is indexes of target sentence (np.ndarray)
            # tuple: tuple[3] is indexes of target sentence mask (np.ndarray)
            # tuple: tuple[4] is dict [0, 3, 4, 2, 29999]   # no duplicated word
            # their shape: (batch_size * sentence_length)
            epoch_start = time.time()
            eidx = epoch + 1
            sys.stderr.write(
                '....................... Epoch [{} / {}] .......................\n'.format(
                    eidx, max_epochs)
            )
            n_samples = 0
            batch_count_in_cur_epoch = 0
            tr_epoch_mean_cost = 0.
            for tr_data in tr_stream.get_epoch_iterator():  # tr_data is a tuple  update one time for one batch

                batch_count += 1
                batch_count_in_cur_epoch += 1

                bx, bxm, by, bym, btvob = tr_data[0], tr_data[
                    1], tr_data[2], tr_data[3], tr_data[4]
                n_samples += len(bx)
                minibatch_size = by.shape[0]

                if config['use_mv']:
                    map_batchno_vcabset = collections.defaultdict(set)
                    for l in btvob:
                        map_batchno_vcabset[0] |= set(l)
                    # small, do not write into file
                    map_batchno_vcabset[0] |= set(ltopk_trg_vocab_idx)
                    batch_sub_vocab = np.unique(
                        np.array(sorted(map_batchno_vcabset[0])).astype('int64'))

                renew_cg()
                nwk.prepare_params()

                nwk_start = time.time()
                batch_error = nwk.get_loss(bx, bxm, by, bym)
                t_nwk = time.time() - nwk_start

                fwd_start = time.time()
                batch_error_val = batch_error.scalar_value()
                t_fwd = time.time() - fwd_start

                tr_epoch_mean_cost += batch_error_val

                bwd_start = time.time()
                batch_error.backward()
                t_bwd = time.time() - bwd_start

                upd_start = time.time()
                nwk.trainer.update()
                t_upd = time.time() - upd_start

                sys.stderr.write(
                    'batch:{}, sent_len:{}, nwk:{}s fwd:{}s, bwd:{}s, upd:{}s\n'.format(
                        batch_count_in_cur_epoch, by.shape[
                            1], format(t_nwk, '0.3f'),
                        format(t_fwd, '0.3f'), format(t_bwd, '0.3f'), format(t_upd, '0.3f')))

                ud = time.time() - fwd_start

                if batch_count % config['display_freq'] == 0:

                    runtime = (time.time() - start_time) / 60.
                    ref_sent_len = by.shape[1]
                    ref_wcnt_wopad = np.count_nonzero(bym)
                    ws_per_sent = ref_wcnt_wopad / minibatch_size
                    sec_per_sent = ud / minibatch_size

                    sys.stderr.write(
                        '[epoch {:>2}]  '
                        '[batch {: >5}]  '
                        '[samples {: >7}]  '
                        '[sent-level loss=>{: >8}]  '
                        '[words/s=>{: >4}/{: >2}={:>6}]  '
                        '[upd/s=>{:>6}/{: >2}={: >5}s]  '
                        '[subvocab {: >4}]  '
                        '[elapsed {:.3f}m]\n'.format(
                            eidx,
                            batch_count_in_cur_epoch,
                            n_samples,
                            format(batch_error_val, '0.3f'),
                            ref_wcnt_wopad, minibatch_size, format(
                                ws_per_sent, '0.3f'),
                            format(ud, '0.3f'), minibatch_size, format(
                                sec_per_sent, '0.3f'),
                            len(batch_sub_vocab) if batch_sub_vocab is not None else 0,
                            runtime)
                    )

                if batch_count % config['sampling_freq'] == 0:
                    renew_cg()
                    nwk.prepare_params()
                    if sample_src_np is not None:
                        trans_samples(sample_src_np, sample_trg_np, nwk, sample_switch, src_vocab_i2w,
                                      trg_vocab_i2w, bos_id=0, eos_id=teos_idx, k=beam_size, mode=search_mode, ptv=batch_vcb_fix)
                    else:
                        trans_samples(bx[:sample_size], by[:sample_size], nwk, sample_switch, src_vocab_i2w,
                                      trg_vocab_i2w, bos_id=0, eos_id=teos_idx, k=beam_size, mode=search_mode, ptv=batch_sub_vocab)

                # sample, just take a look at the translate of some source
                # sentences in training data
                if config['if_fixed_sampling'] and batch_count == batch_start_sample:
                    # select k sample from current batch
                    # rand_rows = random.sample(xrange(batch_size), sample_size)
                    rand_rows = np.random.choice(
                        batch_size, sample_size, replace=False)
                    # randomly select sample_size number from batch_size
                    # rand_rows = np.random.randint(batch_size, size=sample_size)   #
                    # np.int64, may repeat
                    sample_src_np = np.zeros(
                        shape=(sample_size, bx.shape[1])).astype('int64')
                    sample_trg_np = np.zeros(
                        shape=(sample_size, by.shape[1])).astype('int64')
                    for id in xrange(sample_size):
                        sample_src_np[id, :] = bx[rand_rows[id], :]
                        sample_trg_np[id, :] = by[rand_rows[id], :]

                    if config['use_mv']:
                        m_batch_vcb_fix = collections.defaultdict(set)
                        for l in btvob:
                            m_batch_vcb_fix[0] |= set(l)
                        m_batch_vcb_fix[0] |= set(ltopk_trg_vocab_idx)
                        batch_vcb_fix = np.unique(
                            np.array(sorted(m_batch_vcb_fix[0])).astype('int64'))

            mean_cost_on_tr_data = tr_epoch_mean_cost / batch_count_in_cur_epoch
            epoch_time_consume = time.time() - epoch_start
            sys.stderr.write('End epoch [{}], average cost on all training data: {}, consumes time:'
                             '{}s\n'.format(eidx, mean_cost_on_tr_data, format(epoch_time_consume, '0.3f')))
            '''
            # translate dev
            val_time += 1
            sys.stderr.write(
		'Batch [{}], valid time [{}], save model ...\n'.format(batch_count, val_time))

            # save models: search_model_ch2en/params_e5_upd3000.npz
            model_name = '{}_e{}_upd{}.{}'.format(
		config['model_prefix'], eidx, batch_count, 'npz')
            trans.savez(model_name)
            sys.stderr.write(
		'start decoding on validation data [{}]...\n'.format(config['val_set']))
            renew_cg()
            nwk.prep_params()
            cmd = ['sh trans.sh {} {} {} {} {} {} {} {} {} {}'.format(
		eidx,
		batch_count,
		model_name,
		search_mode,
		beam_size,
		config['use_norm'],
		config['use_batch'],
		config['use_score'],
		1,
		config['use_mv'])
            ]
            child = subprocess.Popen(cmd, shell=True)
            '''

        tr_time_consume = time.time() - start_time
        sys.stderr.write('Training consumes time: {}s\n'.format(
            format(tr_time_consume, '0.3f')))
