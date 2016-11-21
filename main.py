from __future__ import division

import os
import sys

from utils import init_dirs
from stream_with_dict import get_tr_stream, ensure_special_tokens
import configurations
from cp_sample import trans_samples
import cPickle as pickle

import subprocess

import numpy as np
import time
import collections
from network import Network

from _gdynet import *
print
# from _dynet import *


def train(
    swemb_dims,
    twemb_dims,
    enc_hidden_units,
    dec_hidden_units,
    align_dims,
    logistic_in_dims,
    src_vocab_size,
    trg_vocab_size,
    droprate,
):
    start_time = time.time()

    nwk = Network(
        swemb_dims=swemb_dims,
        twemb_dims=twemb_dims,
        enc_hidden_units=enc_hidden_units,
        dec_hidden_units=dec_hidden_units,
        align_dims=align_dims,
        logistic_in_dims=logistic_in_dims,
    )
    nwk.init_params()

    for k, v in nwk.params.iteritems():
        sys.stderr.write('    {:20}: {}\n'.format(k, v.shape()))

    k_batch_start_sample = config['k_batch_start_sample']
    batch_size, sample_size = config['batch_size'], config['hook_samples']
    if batch_size < sample_size:
        sys.stderr.write('batch size must be great or equal with sample size')
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
                    batch_count_in_cur_epoch, by.shape[1], format(t_nwk, '0.3f'),
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


if __name__ == "__main__":

    config = getattr(configurations, 'get_config_cs2en')()

    from manvocab import topk_target_vcab_list
    ltopk_trg_vocab_idx = topk_target_vcab_list(**config)

    sys.stderr.write('\nload source and target vocabulary ...\n')
    sys.stderr.write('want to generate source dict {} and target dict {}: \n'.format(
        config['src_vocab_size'], config['trg_vocab_size']))
    src_vocab = pickle.load(open(config['src_vocab']))
    trg_vocab = pickle.load(open(config['trg_vocab']))
    # for k, v in src_vocab.iteritems():
    #	print k, v
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

    train(
        swemb_dims=config['swemb_dims'],
        twemb_dims=config['twemb_dims'],
        enc_hidden_units=config['enc_hidden_units'],
        dec_hidden_units=config['dec_hidden_units'],
        align_dims=config['align_dims'],
        logistic_in_dims=config['logistic_in_dims'],
        src_vocab_size=config['src_vocab_size'],
        trg_vocab_size=config['trg_vocab_size'],
        droprate=config['droprate'],
    )
