from __future__ import division

import os
import sys
import time
import collections
import subprocess
import logging
import cPickle as pickle
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

import numpy
import configurations
from utils import adadelta, step_clipping, init_dirs
from stream_with_dict import get_tr_stream, get_dev_stream, ensure_special_tokens
from sample import trans_sample, multi_process_sample, valid_bleu
#from nmt import Translator
from nmt_debug import Translator


if __name__ == "__main__":
    config = getattr(configurations, 'get_config_cs2en')()

    # prepare data
    logger.info('prepare data ...')
    prepare_file = config['prepare_file']
    subprocess.check_call(" python {}".format(prepare_file), shell=True)

    from manvocab import topk_target_vcab_list
    ltopk_trg_vocab_idx = topk_target_vcab_list(**config)

    logger.info('\tload source and target vocabulary ...')
    logger.info('\twant to generate source dict {} and target dict {}: '.format(
        config['src_vocab_size'], config['trg_vocab_size']))
    src_vocab = pickle.load(open(config['src_vocab']))
    trg_vocab = pickle.load(open(config['trg_vocab']))
    # for k, v in src_vocab.iteritems():
    #	print k, v
    logger.info('\t~done source vocab count: {}, target vocab count: {}'.format(
        len(src_vocab), len(trg_vocab)))
    logger.info('\tvocabulary contains <S>, <UNK> and </S>')

    seos_idx, teos_idx = config['src_vocab_size'] - 1, config['trg_vocab_size'] - 1
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

    tr_stream = get_tr_stream(**config)

    k_batch_start_sample = config['k_batch_start_sample']
    batch_size, sample_size = config['batch_size'], config['hook_samples']
    if batch_size < sample_size:
        logger.info('batch size must be great or equal with sample size')
        sys.exit(0)

    # from numpy.random import RandomState
    # prng = RandomState(1234)
    # batch_start_sample = prng.randint(2, k_batch_start_sample)  # [low, high)
    batch_start_sample = numpy.random.randint(2, k_batch_start_sample)  # [low, high)
    logger.info('batch_start_sample: {}'.format(batch_start_sample))

    batch_count, sent_count, val_time, best_score = 0, 0, 0, 0
    model_name = ''
    sample_src_np, sample_trg_np = None, None

    translator = Translator(**config)

    max_epochs = config['max_epoch']

    logger.info('Start training!!!')
    train_start = dis_start_time = time.time()
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
        n_samples, epoch_total_loss, batch_count_in_cur_epoch = 0, 0., 0
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
            b_loss_sum, minibatch_size, ref_wcnt_wopad = translator.upd_loss_wbatch_with_atten_sbs(
                bx, bxm, by, bym)
            vb_loss_sum = b_loss_sum.value()
            epoch_total_loss += vb_loss_sum
            b_loss_sum.backward()
            translator.trainer.update()
            ud = time.time() - ud_start
            w_per_sent = ref_wcnt_wopad / minibatch_size
            loss_per_word = vb_loss_sum / ref_wcnt_wopad
            sec_per_sent = ud / minibatch_size
            logger.info('avg_loss/word => [{0: >10}/{0: >4} = {0: >6}], '
                        'upd[{0: >6}s], '
                        'avg_words/sent => [{0: >4}/{} = {0: >6}], '
                        'avg_upd/sent => [{0: >6}/{} = {0: >5}s]'.format(
                            format(vb_loss_sum, '0.3f'), ref_wcnt_wopad,
                            format(loss_per_word, '0.3f'),
                            format(ud, '0.3f'),
                            ref_wcnt_wopad, minibatch_size, format(w_per_sent, '0.3f'),
                            format(ud, '0.3f'), minibatch_size, format(sec_per_sent, '0.3f')))

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
                logger.info('source sentence, type: {}, shape: {}'.format(
                    type(nd_source), nd_source.shape))

            if batch_count % config['display_freq'] == 0:
                dis_ud_time = time.time() - dis_start_time
                dis_start_time = time.time()
                logger.info('Epoch[{}], Update(batch)[{}], Seen[{}] samples, subvocab size[{}],'
                            'Loss[{}], Update time[{}s], [{}s] for this {} batchs'.format(
                                eidx, batch_count, n_samples, len(batch_sub_vocab),
                                format(vb_loss_sum, '0.3f'), format(ud, '0.3f'),
                                format(dis_ud_time, '0.3f'), config['display_freq']))

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

        mean_cost_on_tr_data = epoch_total_loss / batch_count_in_cur_epoch
        epoch_time_consume = time.time() - epoch_start
        logger.info('End epoch [{}], average cost on all training data: {}, consumes time: {}s'.format(
            eidx, mean_cost_on_tr_data, format(epoch_time_consume, '0.3f')))
        # translate dev
        val_time += 1
        logger.info('Batch [{}], valid time [{}], save model ...'.format(batch_count, val_time))
        # save models: models_ch2en/params_e5_upd3000.npz
        model_name = '{}_e{}_upd{}.{}'.format(config['model_prefix'], eidx, batch_count, 'npz')
        trans.savez(model_name)
        logger.info('start decoding on validation data [{}]...'.format(config['val_set']))
        child = subprocess.Popen('sh trans.sh {} {} {}'.format(
            eidx, batch_count, model_name), shell=True)

    tr_time_consume = time.time() - train_start
    logger.info('Training consumes time: {}s'.format(format(tr_time_consume, '0.3f')))
