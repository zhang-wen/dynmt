from __future__ import division

import theano
import theano.tensor as T
import os
import sys

from utils import adadelta, step_clipping, init_dirs
from stream_with_dict import get_tr_stream, get_dev_stream, ensure_special_tokens
import logging
import configurations
from sample import trans_sample, multi_process_sample, valid_bleu
import cPickle as pickle
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from nmt import Translator
import subprocess

import numpy
import time
import collections
#import _dynet as dy
import _gdynet as dy

# Get the arguments

if __name__ == "__main__":
    train_start = time.time()
    config = getattr(configurations, 'get_config_cs2en')()

    # prepare data
    logger.info('prepare data ...')
    prepare_file = config['prepare_file']
    subprocess.check_call(" python {}".format(prepare_file), shell=True)
    logger.info('\tload source and target vocabulary ...')
    src_vocab = pickle.load(open(config['src_vocab']))
    trg_vocab = pickle.load(open(config['trg_vocab']))
    src_vocab_num = len(src_vocab)  # unique words number
    trg_vocab_num = len(trg_vocab)  # unique words number
    # for k, v in src_vocab.iteritems():
    #	print k, v
    logger.info('\tsource vocab number: {}'.format(src_vocab_num))
    logger.info('\ttarget vocab number: {}'.format(trg_vocab_num))
    logger.info('\tsource want to generate: {}'.format(config['src_vocab_size']))
    logger.info('\ttarget want to generate: {}'.format(config['trg_vocab_size']))
    logger.info('\tvocabulary contains <S>, <UNK> and </S>')

    # eosid = (src_vocab_num - 1) if src_vocab_num - 3 < config['src_vocab_size'] else (config['src_vocab_size'] + 3 - 1)
    # print 'eos idx in source: ' + str(eosid)

    seos_idx, teos_idx = config['src_vocab_size'] - 1, config['trg_vocab_size'] - 1
    src_vocab = ensure_special_tokens(src_vocab,
                                      bos_idx=0, eos_idx=seos_idx,
                                      unk_idx=config['unk_id'])
    trg_vocab = ensure_special_tokens(trg_vocab,
                                      bos_idx=0, eos_idx=teos_idx,
                                      unk_idx=config['unk_id'])

    # for word, index in src_vocab.iteritems():
    #	    print index, word

    # the trg_vocab is originally:
    #   {'UNK': 1, '<s>': 0, '</s>': 0, 'is': 5, ...}
    # after ensure_special_tokens, the trg_vocab becomes:
    #   {'<UNK>': 1, '<S>': 0, '</S>': trg_vocab_size-1, 'is': 5, ...}
    trg_vocab_i2w = {index: word for word, index in trg_vocab.iteritems()}
    src_vocab_i2w = {index: word for word, index in src_vocab.iteritems()}
    # after reversing, the trg_vocab_i2w become:
    #   {1: '<UNK>', 0: '<S>', trg_vocab_size-1: '</S>', 5: 'is', ...}
    logger.info('load dict finished ! src dic size : {} trg dic size : {}.'.format(
        len(src_vocab), len(trg_vocab)))

    init_dirs(**config)

    tr_stream = get_tr_stream(**config)
    logger.info('Start training!!!')

    k_batch_start_sample = config['k_batch_start_sample']
    batch_size = config['batch_size']
    sample_size = config['hook_samples']

    if batch_size < sample_size:
        logger.info('batch size must be great or equal with sample size')
        sys.exit(0)

    # from numpy.random import RandomState
    # prng = RandomState(1234)
    # batch_start_sample = prng.randint(2, k_batch_start_sample)  # [low, high)
    batch_start_sample = numpy.random.randint(2, k_batch_start_sample)  # [low, high)
    logger.info('batch_start_sample: {}'.format(batch_start_sample))

    batch_count, sent_count = 0, 0
    val_time = 0
    best_score = 0.
    model_name = ''
    sample_src_np, sample_trg_np = None, None
    dis_start_time = time.time()

    from manvocab import tr_vcabset_to_dict, topk_target_vcab_list
    print 'load top {} ...'.format(config['topk_trg_vocab'])
    ltopk_trg_vocab_idx = topk_target_vcab_list(**config)
    print 'done'

    translator = Translator(**config)

    max_epochs = config['max_epoch']
    for epoch in range(max_epochs):
        # take the batch sizes 3 as an example:
        # tuple: tuple[0] is indexes of source sentence (numpy.ndarray)
            # like array([[0, 23, 3, 4, 29999], [0, 2, 1, 29999], [0, 31, 333, 2, 1, 29999]])
        # tuple: tuple[1] is indexes of source sentence mask (numpy.ndarray)
            # like array([[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]])
        # tuple: tuple[2] is indexes of target sentence (numpy.ndarray)
        # tuple: tuple[3] is indexes of target sentence mask (numpy.ndarray)
        # tuple: tuple[4] is dict [0, 3, 4, 2, 29999]   # no duplicated word
        # their shape: (batch_size * sentence_length)
        epoch_start = time.time()
        eidx = epoch + 1
        print '....................... Epoch [{} / {}] .......................'.format(eidx,
                                                                                       max_epochs)
        n_samples = 0
        batch_count_in_cur_epoch = 0
        epoch_total_loss = 0.
        for tr_data in tr_stream.get_epoch_iterator():  # tr_data is a tuple  update one time for one batch
            batch_count += 1
            batch_count_in_cur_epoch += 1
            n_samples += len(tr_data[0])

            bx, bxm, by, bym, btdic = tr_data[0], tr_data[1], tr_data[2], tr_data[3], tr_data[4]
            map_batchno_vcabset = collections.defaultdict(set)
            for tdic in btdic:
                map_batchno_vcabset[0] |= set(tdic)
            map_batchno_vcabset[0] |= set(ltopk_trg_vocab_idx)  # small, do not write into file
            batch_sub_vocab = numpy.unique(numpy.array(
                list(map_batchno_vcabset[0])).astype('int32'))

            ud_start = time.time()
            bloss, total_loss = 0., 0.
            bloss, minibatch_size, total_reflen_batch = translator.upd_loss_wbatch_with_atten_sbs(
                bx, bxm, by, bym)
            # print 'avg_bloss => ', type(bloss), bloss.npvalue().shape
            vbloss = bloss.value()
            epoch_total_loss += vbloss
            bloss.backward()
            translator.trainer.update()
            ud = time.time() - ud_start
            avg_sentlen = total_reflen_batch / minibatch_size
            sec_per_sent = ud / avg_sentlen
            logger.info('Avg_loss / sent[{} / {}={}],'
                        'Upd[{}s],'
                        'Avg_ref_sents / batch[{} / {}={}],'
                        'Avg_upd / sent[{} / {}={}s]'.format(
                            vbloss, avg_sentlen, vbloss / avg_sentlen,
                            ud,
                            total_reflen_batch, minibatch_size, avg_sentlen,
                            ud, avg_sentlen, sec_per_sent))

            # sample, just take a look at the translate of some source sentences in training data
            if config['if_fixed_sampling'] and batch_count == batch_start_sample:
                # select k sample from current batch
                logger.info('Start fixed sampling, random generate {} from the {}th batch'.format(
                    sample_size, batch_start_sample))
                # list_idx = random.sample(xrange(batch_size), sample_size)
                list_idx = numpy.random.choice(batch_size, sample_size, replace=False)
                # randomly select sample_size number from batch_size
                # list_idx = numpy.random.randint(batch_size, size=sample_size)   #
                # numpy.int64, may repeat
                logger.info('sample sentence indexes in {}th batch: {}'.format(
                    batch_start_sample, list_idx))
                sample_src_np = numpy.zeros(shape=(sample_size, nd_source.shape[1])).astype('int64')
                sample_trg_np = numpy.zeros(shape=(sample_size, nd_target.shape[1])).astype('int64')
                sample_sentnos = []
                for row in xrange(sample_size):
                    sample_src_np[row, :] = nd_source[list_idx[row], :]
                    sample_trg_np[row, :] = nd_target[list_idx[row], :]
                    sample_sentnos.append((batch_count_in_cur_epoch - 1) *
                                          config['batch_size'] + list_idx[row] + 1)     # start from 1
                logger.info('sample source sentence, type: {}, shape: {}'.format(
                    type(sample_src_np), sample_src_np.shape))
                # print sample_src_np
                logger.info('source sentence, type: {}, shape: {}'.format(
                    type(nd_source), nd_source.shape))
                # print nd_source

            if batch_count % config['display_freq'] == 0:
                dis_ud_time = time.time() - dis_start_time
                dis_start_time = time.time()
                logger.info('Epoch[{}], Update(batch)[{}], Seen[{}] samples, subvocab size[{}],'
                            'Loss[{}], Update time[{}s], [{}s] for this {} batchs'.format(
                                eidx, batch_count, n_samples, len(batch_sub_vocab),
                                format(vbloss, '0.3f'), format(ud, '0.3f'),
                                format(dis_ud_time, '0.3f'), config['display_freq']))
                # print type(total_batchno_map_vcbset[batch_count_in_cur_epoch])
                # print total_batchno_map_vcbset[batch_count_in_cur_epoch]

            if batch_count % config['sampling_freq'] == 0:
                logger.info('translation {} samples from {}th batch'.format(
                    sample_size, batch_count))
                if sample_src_np is not None:
                    trans_sample(sample_src_np, sample_trg_np, f_init, f_next, f_next_mv, sample_size, src_vocab_i2w,
                                 trg_vocab_i2w, sample_sentnos, sub_vcbdict=batch_sub_vocab, usebatch_dict=True)
                else:
                    sample_sentnos = range(
                        (batch_count_in_cur_epoch - 1) * config['batch_size'] + 1, batch_count_in_cur_epoch * config['batch_size'] + 1)
                    for i in range(sample_size):
                        src_sent, ref_sent = tr_data[0][i], tr_data[2][i]
                        src_sent = filter(lambda x: x != 0, src_sent)
                        ref_sent = filter(lambda x: x != 0, ref_sent)
                        out = translator.translate(src_sent, ref_sent, teos_idx,
                                                   src_vocab_i2w, trg_vocab_i2w)

            '''
            if batch_count % config['save_freq'] == 0:
                logger.info('epoch [{}], batch [{}], save model ...'.format(eidx, batch_count))
                # save models: search_model_ch2en/params_e5_upd3000.npz
                model_name = '{}_e{}_upd{}.{}'.format(
                    config['model_prefix'], eidx, batch_count, 'npz')
                trans.savez(model_name)

            # trans validation data set
            if batch_count > config['val_burn_in'] and batch_count % config['bleu_val_freq'] == 0:
                if not os.path.exists(model_name):
                    logger.info(
                        'decoding on validation set before saving model ... please wait model saved')
                else:
                    logger.info('epoch [{}], batch [{}]: start decoding on validation data [{}]...'.format(
                        eidx, batch_count, config['val_set']))
                    val_time += 1
                    child = subprocess.Popen('sh trans.sh {} {} {}'.format(
                        eidx, batch_count, model_name), shell=True)
            '''

        mean_cost_on_tr_data = epoch_total_loss / batch_count_in_cur_epoch
        epoch_time_consume = time.time() - epoch_start
        logger.info('End epoch [{}], average cost on all training data: {}, consumes time: {}s'.format(
            eidx, mean_cost_on_tr_data, format(epoch_time_consume, '0.3f')))
        # translate dev
        val_time += 1
        logger.info('Batch [{}], valid time [{}], save model ...'.format(batch_count, val_time))
        # save models: search_model_ch2en/params_e5_upd3000.npz
        model_name = '{}_e{}_upd{}.{}'.format(config['model_prefix'], eidx, batch_count, 'npz')
        trans.savez(model_name)
        logger.info('start decoding on validation data [{}]...'.format(config['val_set']))
        child = subprocess.Popen('sh trans.sh {} {} {}'.format(
            eidx, batch_count, model_name), shell=True)

    tr_time_consume = time.time() - train_start
    logger.info('Training consumes time: {}s'.format(format(tr_time_consume, '0.3f')))
