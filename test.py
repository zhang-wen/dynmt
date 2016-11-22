import numpy
import copy
from functools import partial
from multiprocessing import Pool
import os
import subprocess
import re
from utils import exeTime
import collections

import logging
logger = logging.getLogger('sample')
logging.basicConfig(level=logging.INFO)

from numpy import unique


def _index2sentence(vec, dic, part=None):
    if part is not None:
        # OrderedDict([(0, 0), (1, 1), (3, 5), (8, 2), (10, 3), (100, 4)])
        # reverse: OrderedDict([(0, 0), (1, 1), (5, 3), (2, 8), (3, 10), (4, 100)])
        # part[index] get the real index in large target vocab firstly
        r = [dic[part[index]] for index in vec]
    else:
        r = [dic[index] for index in vec]
    r = " ".join(r)
    r = r + ' ' if r.endswith('@@') else r
    return r


def part_sort(vec, num):
    # numpy.argpartition(vec, num)
    # put the num small data before the num_th position
    # get the index of the num small data numpy.argpartition(vec, num)[:num]
    #  The kth element will be in its final sorted position and all smaller elements will be moved
    #  before it and all larger elements behind it. The order all elements in the partitions is
    #  undefined.
    ind = numpy.argpartition(vec, num)[:num]
    # sort the small data before the num_th element, and get the index
    return ind[numpy.argsort(vec[ind])]


# part_trg_idset_in_large_vcb is a sub-dict of large dict
def gen_sample(x, f_init, f_next, f_next_mv, k=10, maxlen=40, vocab=None, normalize=True,
               part_trg_idset_in_large_vcb=None, valid=False, ifmv=False):
    if valid is True:
        part_trg_idset_in_large_vcb = unique(numpy.asarray(
            x[1], dtype='int32'))  # subdict set [0,2,6,29999, 333]
        sent = x[0]
    else:
        sent = x

    # k is the beam size we have
    sent = numpy.asarray(sent, dtype='int64')
    if sent.ndim == 1:  # x (5,)
        sent = sent[None, :]  # x(1, 5)
    maxlen = sent.shape[1] * 2     # x(batch_size, src_sent_len)
    sent = sent.T
    eos_id = len(vocab) - 1
    bos_id = 0

    sample = []
    sample_score = []

    live_k = 1
    dead_k = 0

    hyp_samples = [[]] * live_k
    hyp_scores = numpy.zeros(live_k).astype('float32')
    hyp_states = []

    # get initial state of decoder rnn and encoder context
    ret = f_init(sent)
    next_state, ctx0 = ret[0], ret[1]
    next_w = [-1]  # indicator for the first target word (bos target)

    for ii in xrange(maxlen):
        ctx = numpy.tile(ctx0, [live_k, 1])
        #print 'part_trg_idset_in_large_vcb:'
        #print part_trg_idset_in_large_vcb
        # part_trg_idset_in_large_vcb: [0, 1, 3, 23]

        if ifmv:
            inps = [next_w, ctx, next_state, part_trg_idset_in_large_vcb]
            ret = f_next_mv(*inps)
        else:
            inps = [next_w, ctx, next_state]
            ret = f_next(*inps)
        next_p, next_state = ret[0], ret[1]

        cand_scores = hyp_scores[:, None] - numpy.log(next_p)
        cand_flat = cand_scores.flatten()
        #ranks_flat = cand_flat.argsort()[:(k-dead_k)]
        # the less, the better, faster ... partition first then sort, no need to sort all
        ranks_flat = part_sort(cand_flat, k - dead_k)

        voc_size = next_p.shape[1]  # size of sub dict
        trans_indices = ranks_flat / voc_size
        word_indices = ranks_flat % voc_size
        '''
        print 'voc_size:'
        print voc_size
        print 'trans_indices: '
        print trans_indices
        print 'word_indices: '
        print word_indices
        '''
        costs = cand_flat[ranks_flat]

        new_hyp_samples = []
        new_hyp_scores = numpy.zeros(k - dead_k).astype('float32')
        new_hyp_states = []

        for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
            new_hyp_samples.append(hyp_samples[ti] + [wi])
            new_hyp_scores[idx] = copy.copy(costs[idx])
            new_hyp_states.append(copy.copy(next_state[ti]))

        # print 'new_hyp_samples: '
        # print new_hyp_samples

        # check the finished samples
        new_live_k = 0
        hyp_samples = []
        hyp_scores = []
        hyp_states = []

        # part_trg_vocab_nid2oid: {0:0, 1:1, 2:3, 3:23}
        for idx in xrange(len(new_hyp_samples)):
            # such as: part_trg_vocab_nid2oid[34] == 29999
            tmpwid = part_trg_idset_in_large_vcb[new_hyp_samples[
                idx][-1]] if ifmv else new_hyp_samples[idx][-1]
            if tmpwid == eos_id:
                sample.append(new_hyp_samples[idx])
                sample_score.append(new_hyp_scores[idx])
                dead_k += 1
            else:
                new_live_k += 1
                hyp_samples.append(new_hyp_samples[idx])
                hyp_scores.append(new_hyp_scores[idx])
                hyp_states.append(new_hyp_states[idx])
        hyp_scores = numpy.array(hyp_scores)
        live_k = new_live_k

        if new_live_k < 1:
            break
        if dead_k >= k:
            break

        next_w = numpy.array([w[-1] for w in hyp_samples])
        next_state = numpy.array(hyp_states)

    if live_k > 0:
        for idx in xrange(live_k):
            sample.append(hyp_samples[idx])
            sample_score.append(hyp_scores[idx])

    if normalize:
        lengths = numpy.array([len(s) for s in sample])
        sample_score = sample_score / lengths
    sidx = numpy.argmin(sample_score)

    best_trans = sample[sidx]
    # index in best_trans is sub_index, not the real index in large vocabulary
    # print best_trans
    # print '-----------------'
    if ifmv:
        best_trans = filter(lambda item: part_trg_idset_in_large_vcb[
                            item] != eos_id and part_trg_idset_in_large_vcb[item] != bos_id, best_trans)
    else:
        best_trans = filter(lambda item: item != eos_id and item != bos_id, best_trans)
    if vocab is not None:
        if ifmv:
            # print best_trans
            # print part_trg_vocab_nid2oid
            # part_trg_vocab_nid2oid: {0:0, 1:1, 2:3, 3:23}
            best_trans = _index2sentence(best_trans, vocab, part_trg_idset_in_large_vcb)
        else:
            best_trans = _index2sentence(best_trans, vocab)
    return best_trans


def trans_sample(s, t, f_init, f_next, f_next_mv, hook_samples, src_vocab_i2w, trg_vocab_i2w,
                 sample_sentnos=None, sub_vcbdict=None, usebatch_dict=True):
    hook_samples = min(hook_samples, s.shape[0])
    eos_id_src = len(src_vocab_i2w) - 1
    eos_id_trg = len(trg_vocab_i2w) - 1
    for index in range(hook_samples):
        # print 'before filter: '
        # print s[index]
        # [   37   785   600    44   160  4074   152  3737     2   399  1096   170      4     8    29999     0     0     0     0     0     0     0]
        s_filter = filter(lambda x: x != 0, s[index])
        # s_filter: [   37   785   600    44   160  4074   152  3737     2   399
        # 1096   170      4 8    29999]
        sent_number = sample_sentnos[index]
        # print 'sent_number:'
        # print sent_number
        if usebatch_dict:
            # np_dict = np.asarray([0, 1, 2, 3, 5], dtype=np.int32)
            sentno_trg_dict = sub_vcbdict
        else:
            sentno_trg_dict = sub_vcbdict[sent_number]
        # print 'sentno_trg_dict:'
        #{0:0,1:1,3:2,23:3}
        # print sentno_trg_dict
        '''
        np_sent_level_vocab_set = numpy.zeros((len(sentno_trg_dict),)).astype('int32')
        part_trg_vocab_nid2oid = {}
        #sorted_sent_trg_dict = collections.OrderedDict(sorted(sentno_trg_dict.items()))
        for k, v in sentno_trg_dict.iteritems():
            np_sent_level_vocab_set[v] = k  # [0, 1, 3, 23]
            part_trg_vocab_nid2oid[v] = k   # {0:0, 1:1, 2:3, 3:23}
        '''

        logger.info('[{:3}] {}'.format('src', _index2sentence(s_filter, src_vocab_i2w)))
        ref = _index2sentence(filter(lambda x: x != 0, t[index]), trg_vocab_i2w)
        logger.info('[{:3}] {}'.format('ref', ref))
        trans = gen_sample(s_filter, f_init, f_next, f_next_mv, k=2, vocab=trg_vocab_i2w,
                           part_trg_idset_in_large_vcb=unique(sentno_trg_dict), ifmv=True)
        logger.info('[{:3}] {}\n'.format('out', trans))


@exeTime
def multi_process_sample(x_iter, f_init, f_next, f_next_mv, k=10, maxlen=50, trg_vocab_i2w=None, normalize=True, process=5):
    partial_func = partial(gen_sample, f_init=f_init, f_next=f_next, f_next_mv=f_next_mv, k=k,
                           maxlen=maxlen, vocab=trg_vocab_i2w, normalize=normalize, valid=True,
                           ifmv=True)
    if process > 1:
        pool = Pool(process)
        trans_res = pool.map(partial_func, x_iter)
    else:
        trans_res = map(partial_func, x_iter)

    trans_res = ['{}\n'.format(item) for item in trans_res]
    return trans_res


def fetch_bleu_from_file(fbleufrom):
    fread = open(fbleufrom, 'r')
    result = fread.readlines()
    fread.close()
    f_bleu = 0.
    f_multibleu = 0.
    for line in result:
        bleu_pattern = re.search(r'BLEU score = (0\.\d+)', line)
        if bleu_pattern:
            s_bleu = bleu_pattern.group(1)
            f_bleu = format(float(s_bleu) * 100, '0.2f')
        multi_bleu_pattern = re.search(r'BLEU = (\d+\.\d+)', line)
        if multi_bleu_pattern:
            s_multibleu = multi_bleu_pattern.group(1)
            f_multibleu = format(float(s_multibleu), '0.2f')
    return f_bleu, f_multibleu


def append_file(file_prefix, content):
    f = open(file_prefix, 'a')
    f.write(content)
    f.write('\n')
    f.close()


def valid_bleu(eval_dir, valid_out, eidx, uidx):    # valid_out: valids/trans
    import configurations
    config = getattr(configurations, 'get_config_cs2en')()
    save_log = '{}.{}'.format(valid_out, 'log')
    #oriref_bleu_log = '{}_e{}_upd{}.{}'.format(valid_out, eidx, uidx, 'prolog')
    # child = subprocess.Popen('sh wztrans.sh ../{} {} ../{} ../{} {}'.format(valid_out,
    #                                                                        config['val_prefix'],
    #                                                                        save_log, oriref_bleu_log,
    #                                                                        config['val_tst_dir']),
    #                         cwd=eval_dir, shell=True, stdout=subprocess.PIPE)
    # print 'sh bleu.sh ../{} {} ../{} {}'.format(valid_out,config['val_prefix'], save_log,
    #                                            config['val_tst_dir'])
    child = subprocess.Popen('sh bleu.sh ../{} {} ../{} {}'.format(valid_out, config['val_prefix'],
                                                                   save_log, config['val_tst_dir']),
                             cwd=eval_dir, shell=True, stdout=subprocess.PIPE)
    bleu_out = child.communicate()
    child.wait()
    mteval_bleu, multi_bleu = fetch_bleu_from_file(save_log)
    #ori_mteval_bleu, ori_multi_bleu = fetch_bleu_from_file(oriref_bleu_log)
    sfig = '{}.{}'.format(config['val_set_out'], 'sfig')
    # sfig_content = str(eidx) + ' ' + str(uidx) + ' ' + str(mteval_bleu) + ' ' + \
    #    str(multi_bleu) + ' ' + str(ori_mteval_bleu) + ' ' + str(ori_multi_bleu)
    sfig_content = str(eidx) + ' ' + str(uidx) + ' ' + str(mteval_bleu) + ' ' + str(multi_bleu)
    append_file(sfig, sfig_content)

    os.rename(valid_out, "{}_{}_{}.txt".format(valid_out, mteval_bleu, multi_bleu))
    return mteval_bleu  # we use the bleu without unk and mteval-v11, process reference

if __name__ == "__main__":
    import sys
    res = valid_bleu(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    print res
