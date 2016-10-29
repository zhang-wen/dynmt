# -*- coding: utf-8 -*-

import numpy
import logging
from itertools import izip
import time

logger = logging.getLogger('manvocab')
logging.basicConfig(level=logging.INFO)

import pickle
from picklable_itertools.extras import equizip
import collections
from utils import exeTime
from stream_with_dict import get_tr_stream

import collections
import os


@exeTime
def trg_cands_from_aln(src, trg, aln, src_vocab, trg_vocab):
    trg_cands_dict = collections.defaultdict(list)
    lineno = 0
    try:
        with open(src, 'r') as fsrc, open(trg, 'r') as ftrg, open(aln, 'r') as faln:
            for lsrc, ltrg, laln in equizip(fsrc, ftrg, faln):
                list_src, list_trg, list_aln = None, None, None
                if (lsrc != '\n') and (ltrg != '\n') and (laln != '\n'):
                    lineno += 1
                    if lineno % 100000 == 0:
                        print lineno
                    list_src = lsrc.split()
                    list_trg = ltrg.split()  # i am a good
                    list_aln = laln.split()  # 0-1 1-0
                    for item in list_aln:
                        sid_insent = int(item.split('-')[0])
                        tid_insent = int(item.split('-')[1])
                        src_word = list_src[sid_insent]
                        trg_word = list_trg[tid_insent]
                        sid_indict = 1 if src_word not in src_vocab else src_vocab[src_word]
                        tid_indict = 1 if trg_word not in trg_vocab else trg_vocab[trg_word]
                        # the target alignment is too much for UNK, do not consider
                        if not sid_indict == 1 and not tid_indict == 1:
                            if tid_indict not in trg_cands_dict[sid_indict]:
                                trg_cands_dict[sid_indict].append(tid_indict)
    except IOError:
        print 'may be no some file, open failed.'
    # defaultdict(list, {2: [3, 4], 4: [0]})    {src_id_in_dict: [trg_id_in_dict0, ..., ]}
    return trg_cands_dict


def _2dnumpy2list(ndarr):
    ref_vocab = []
    for sent in ndarr:
        for tword_idx in sent:
            ref_vocab.append(tword_idx)
    return list(set(ref_vocab))


def append_file(file_prefix, content):
    f = open(file_prefix, 'a')
    f.write(content)
    f.write('\n')
    f.close()


# src_vocab: {我:3, 爱:4, 你:22}
def load_lexical_trans_table(src_vocab=None, trg_vocab=None, lex_trans_table_file=None, ifword=False):
    if ifword:
        logger.info('load lexical translation table (word) from {}'.format(lex_trans_table_file))
    else:
        logger.info('load lexical translation table (index) from {}'.format(lex_trans_table_file))

    trg_cands_dict = collections.defaultdict(set)
    if lex_trans_table_file is not None:
        with open(lex_trans_table_file, 'r') as flex:
            lineno = 0
            for word_pair in flex:
                lineno += 1
                if lineno % 1000000 == 0:
                    logger.info(lineno)
                trg_src_prob = word_pair.split(' ')
                trg_word = trg_src_prob[0]  # en
                src_word = trg_src_prob[1]  # zh
                prob = trg_src_prob[2]
                # note here, prob is string, so need convert, otherwise '0.0001' > 0.1
                if float(prob) < 0.1:
                    continue     # filter lex
                # if not sid_indict == 1 and not tid_indict == 1:
                if ifword:
                    trg_cands_dict[src_word].add(trg_word)
                else:
                    sid_indict = 1 if src_word not in src_vocab else src_vocab[src_word]
                    tid_indict = 1 if trg_word not in trg_vocab else trg_vocab[trg_word]
                    trg_cands_dict[sid_indict].add(tid_indict)
    logger.info('done')
    return trg_cands_dict


def load_phrase_table(src_vocab, trg_vocab, phrase_table_file=None):
    logger.info('load phrase table from {}'.format(phrase_table_file))
    trg_phrase_cands = collections.defaultdict(set)
    if phrase_table_file:
        with gzip.open(phrase_table_file, 'rb') as f:
            lineno = 0
            for line in f:
                # ! [NP][X] [PRN] ||| ! [NP][X] [X] ||| 0.115896 0.815504 0.0289739 0.662052 ||| 0-0 1-1
                # ||| 0.2 0.8 0.2 ||| |||
                lineno += 1
                if lineno % 1000000 == 0:
                    logger.info(lineno)
                rules = line.split('|||')
                source = rules[0]
                target = rules[1]
                scores = rules[2]
                p_e_give_f = scores.split(' ')[2]
                if p_e_give_f < 0.1:
                    continue   # filter rules
                list_key = []
                # 我 爱 你  ->  [3, 4, 22] -> '3 4 22' as key
                for src_word in source.split(' '):
                    sid_indict = 1 if src_word not in src_vocab else src_vocab[src_word]
                    list_key.append(sid_indict)
                src_phrase_key = ' '.join(map(str, list_key))
                for trg_word in target.split(' '):
                    tid_indict = 1 if trg_word not in trg_vocab else trg_vocab[trg_word]
                    # if not tid_indict == 1:
                    trg_phrase_cands[src_phrase_key].add(tid_indict)
    logger.info('done')
    return trg_phrase_cands

import gzip


def load_trg_vocab_sent(trg_cands_dict_per_sent, sentno, source, source_mask=None,
                        trg_cands_dict_set=None, trg_phrase_cands_set=None):
    sent_len = len(source) if (source_mask is None) else 0
    if source_mask is not None:
        for wid in xrange(len(source)):
            if source_mask[wid] == 0:
                break
            sent_len += 1   # <s> good morning </s>

    # V_x^D
    for wid in xrange(sent_len):
        src_word_id = source[wid]
        if src_word_id in trg_cands_dict_set:
            trg_cands_dict_per_sent[sentno] |= trg_cands_dict_set[src_word_id]
        # defaultdict(<type 'set'>, {9: set([33, 22, 11, 111]), 100: set([3, 4, 5])})

    # V_x^P
    # search all pharse between 2 and 5
    if trg_phrase_cands_set:
        for phrase_len in xrange(2, 6):
            for i in xrange(sent_len):
                if i + phrase_len <= sent_len:
                    phrase_int_ndarray = source[i: i + phrase_len]   # array([   34,     5, 29999])
                    phrase_str = ' '.join(map(str, phrase))         # '34 5 29999'
                    if phrase_str in trg_phrase_cands_set:
                        trg_cands_dict_per_sent[sentno] |= trg_phrase_cands_set[phrase_str]


def topk_target_vcab_list(topk_trg_pkl, ifword=False, **kwargs):
    # .data/50vocab.zh-en.en.pkl
    # V_x^T
    topk_trg_vocab = pickle.load(open(topk_trg_pkl))  # dictionary
    topk_trg_vocab_num = len(topk_trg_vocab)  # unique words number
    logger.info('\tload top {} target words'.format(topk_trg_vocab_num))
    #src_vocab_reverse = {index: word for word, index in src_vocab.iteritems()}
    ltopk_trg_vocab_idx = []
    for word, index in topk_trg_vocab.iteritems():
        if ifword:
            ltopk_trg_vocab_idx.append(word)
        else:
            ltopk_trg_vocab_idx.append(index)
    return ltopk_trg_vocab_idx


def write_tr_target_word_into_file(src_data, trg_data, dict_data, src_vocab, trg_vocab, **kwargs):
    import configurations
    config = getattr(configurations, 'get_config_cs2en')()

    if os.path.exists(dict_data):
        logger.info('{} exist'.format(dict_data))
        return

    trg_lexical_table = load_lexical_trans_table(
        lex_trans_table_file=config['lex_f2e'], ifword=True)    # dict(set)
    #trg_phrase_table = load_phrase_table(src_vocab, trg_vocab, config['phrase_table'])

    logger.info('start write target sub dict sentence by sentence for training data ...')
    sent_number = 0
    try:
        with open(src_data, 'r') as left, open(trg_data, 'r') as right, open(dict_data, 'w') as dic:
            for src, trg in equizip(left, right):
                sent_number += 1
                if sent_number % 100000 == 0:
                    logger.info('{}'.format(sent_number))
                if (src != '\n') and (trg != '\n'):
                    src = src.strip()
                    trg = trg.strip()
                    map_sentno_vcabset = collections.defaultdict(set)
                    # V_x^D
                    load_trg_vocab_sent(map_sentno_vcabset, 0, source=src.split(),
                                        trg_cands_dict_set=trg_lexical_table)
                    # no V_x^P temparaily
                    # V_x^R
                    ltrg = trg.split()
                    for tw in ltrg:
                        map_sentno_vcabset[0].add(tw)
                    # V_x^T
                    # print line
                    # print map_sentno_vcabset[0]
                    # set(['NULL', 'outstanding', 'very', 'leading', ',', '.', 'this', 'role', 'year', 'still', 'prominent'])
                    # should be shared, do not make the sentence level dict too large
                    #total_set = map_sentno_vcabset[0] | set(topk_trg_vocab_word)
                    # add end start and unk symbols for each sentence target dict for predicting, training data
                    # do not need to add, because each reference have <eos>, we add 0 and 1 forcibly
                    total_set = map_sentno_vcabset[0] | {config['bos_token'],
                                                         config['eos_token'], config['unk_token']}
                    # print total_set
                    # print len(total_set)
                    dic.write(' '.join(list(total_set)) + '\n')  # set no duplicated word
    except IOError:
        logger.error('check some file ...')
    logger.info('traverse training data, sentences number: {}'.format(sent_number))


def tr_batch_target_vcab_set(src_vocab, trg_vocab, topk_trg_vocab_idx):
    import configurations
    config = getattr(configurations, 'get_config_cs2en')()

    trg_lexical_table = load_lexical_trans_table(src_vocab, trg_vocab, config['lex_f2e'])
    #trg_phrase_table = load_phrase_table(src_vocab, trg_vocab, config['phrase_table'])

    train = get_tr_stream(**config)
    logger.info('start load vocabulary for each sentence and batch in training data ...')
    batch_number, sent_number = 0, 0
    import collections
    #map_sentno_vcabset = collections.defaultdict(set)
    map_batchno_vcabset = collections.defaultdict(set)
    for train_data in train.get_epoch_iterator():
        batch_number += 1
        nd_source, nd_source_mask, nd_target, nd_target_mask = train_data[
            0], train_data[1], train_data[2], train_data[3]
        cur_batch_size = len(nd_source)
        for batch_no in xrange(cur_batch_size):
            sent_number += 1
            if sent_number % 100000 == 0:
                logger.info('sent {}'.format(sent_number))
            # load_trg_vocab_sent(map_sentno_vcabset, sent_number, source=nd_source[
            #                    batch_no], source_mask=nd_source_mask[batch_no], trg_cands_dict_set=trg_lexical_table)
            #map_batchno_vcabset[batch_number] |= map_sentno_vcabset[sent_number]
            # print map_sentno_vcabset
            # map_sentno_vcabset is a large dict -> sentence number: set(target vocab)
            load_trg_vocab_sent(map_batchno_vcabset, batch_number, source=nd_source[batch_no],
                                source_mask=nd_source_mask[batch_no], trg_cands_dict_set=trg_lexical_table)
            # V_x^R
            for twid in xrange(len(nd_target[batch_no])):
                if nd_target_mask[batch_no][twid] == 0:
                    break
                # map_sentno_vcabset[sent_number].add(nd_target[batch_no][twid])
                map_batchno_vcabset[batch_number].add(nd_target[batch_no][twid])
            # V_x^T
            #map_sentno_vcabset[sent_number] |= set(topk_trg_vocab_idx)
        map_batchno_vcabset[batch_number] |= set(topk_trg_vocab_idx)
    logger.info('traverse training data, batchs: {}, sentences: {}'.format(batch_number, sent_number))
    # return map_sentno_vcabset, map_batchno_vcabset
    return None, map_batchno_vcabset


def tr_vcabset_to_dict(map_sentno_vcabset=None, map_batchno_vcabset=None):
    from collections import Counter
    map_sentno_vcbdict, map_batchno_vcbdict = None, None
    if map_sentno_vcabset:
        #logger.info('re-mapping index of vocabulary for each sentence ...')
        map_sentno_vcbdict = collections.defaultdict(dict)
        for k, v in map_sentno_vcabset.iteritems():
            counter = Counter(v)
            if 0 in counter:
                counter.pop(0)  # remove 0 and 1 first
            if 1 in counter:
                counter.pop(1)
            vocab_count = counter.most_common()
            vocab = {0: 0, 1: 1}    # unk and </s>
            for i, (word, count) in enumerate(vocab_count):
                vocab[word] = i + 2
            # vocab = ensure_special_tokens(vocab, bos_idx=0, eos_idx=len(
            #    vocab) - 1, unk_idx=config['unk_id'])
            map_sentno_vcbdict[k] = vocab

    if map_batchno_vcabset:
        #logger.info('re-mapping index of vocabulary for each batch ...')
        map_batchno_vcbdict = collections.defaultdict(dict)
        for k, v in map_batchno_vcabset.iteritems():
            counter = Counter(v)
            if 0 in counter:
                counter.pop(0)
            if 1 in counter:
                counter.pop(1)
            vocab_count = counter.most_common()
            vocab = {0: 0, 1: 1}    # unk and </s>
            for i, (word, count) in enumerate(vocab_count):
                vocab[word] = i + 2
            # vocab = ensure_special_tokens(vocab, bos_idx=0, eos_idx=len(
            #    vocab) - 1, unk_idx=config['unk_id'])
            map_batchno_vcbdict[k] = vocab
        # print map_sentno_vcabset
    return map_sentno_vcbdict, map_batchno_vcbdict


def valid_sent_target_vcab_set(src_vocab, lex_f2e, topk_trg_pkl, val_set, valid_sent_dict, bos_token,
                               eos_token, unk_token, **kwargs):
    if os.path.exists(valid_sent_dict):
        logger.info('{} exist'.format(valid_sent_dict))
        return
    logger.info('start load vocabulary for each sentence in validation data and write into file  ...')
    trg_lexical_table = load_lexical_trans_table(lex_trans_table_file=lex_f2e, ifword=True)
    # print 'zhangwen'
    # print trg_lexical_table['毛岸青']
    # print trg_lexical_table['广东省']
    #trg_phrase_table = load_phrase_table(src_vocab, trg_vocab, config['phrase_table'])

    # V_x^T in validation set
    topk_trg_vocab_word = topk_target_vcab_list(topk_trg_pkl, True)

    '''
    array([[0, 1649, 1764, 7458, 1, 29999],
           [0, 174, 210, 1130, 3, 206, 5, 394, 310, 2, 85, 165, 199, 122, 98, 55, 7730, 1803, 5, 90, 55, 206, 504, 1803, 5, 708, 191, 234, 1803, 24, 29999],
           [0, 262, 1753, 1902, 2755, 53, 2, 85, 9237, 95, 4083, 106, 67, 3, 5727, 2, 6828, 918, 1922, 29999]], dtype=object)
    '''
    lines = open(val_set).readlines()
    f_val_sent_dict_set = open(valid_sent_dict, 'w')
    content = []
    for line in lines:
        line = line.strip()
        map_sentno_vcabset = collections.defaultdict(set)
        # V_x^D
        load_trg_vocab_sent(map_sentno_vcabset, 0, source=line.split(),
                            trg_cands_dict_set=trg_lexical_table)
        # no V_x^P temparaily
        # V_x^T
        # print line
        # print map_sentno_vcabset[0]
        # set(['NULL', 'outstanding', 'very', 'leading', ',', '.', 'this', 'role', 'year', 'still', 'prominent'])
        # print set(topk_trg_vocab_word)
        total_set = map_sentno_vcabset[0] | set(topk_trg_vocab_word)
        # add end start and unk symbols for each sentence target dict for predicting, training data
        # do not need to add, because each reference have <eos>, we add 0 and 1 forcibly
        total_set = total_set | {bos_token, eos_token, unk_token}
        # print total_set
        # print len(total_set)
        content.append(' '.join(list(total_set)))
    f_val_sent_dict_set.write('\n'.join(content))
    f_val_sent_dict_set.close()
    logger.info('written dict of each sentence with top words for {} '
                'validation'.format(len(content)))


def load_valid_sent_level_dict(x, src_vocab, trg_vocab):
    import configurations
    config = getattr(configurations, 'get_config_cs2en')()

    trg_lexical_table = load_lexical_trans_table(src_vocab, trg_vocab, config['lex_f2e'])
    # trg_lexical_table: dict(set)

    map_sentno_vcabset = collections.defaultdict(set)
    load_trg_vocab_sent(map_sentno_vcabset, 0, x,
                        trg_cands_dict_set=trg_lexical_table)
    ltopk_trg_vocab_idx = topk_target_vcab_list(config['topk_trg_pkl'])
    map_sentno_vcabset[0] |= set(ltopk_trg_vocab_idx)
    valid_sentno_vcbdict, _ = tr_vcabset_to_dict(map_sentno_vcabset=map_sentno_vcabset)
    #{0:0,1:1,3:2,23:3}
    for k, v in valid_sentno_vcbdict[0].iteritems():
        part_trg_vocab_nid2oid[v] = k
    #{0:0,1:1,2:3,3:23}
    np_sent_level_vocab_set = numpy.zeros((len(valid_sentno_vcbdict[0]),)).astype('int32')
    sorted_batch_trg_dict = collections.OrderedDict(sorted(part_trg_vocab_nid2oid.items()))
    for k, v in sorted_batch_trg_dict.iteritems():
        np_sent_level_vocab_set[k] = v
    part_target_vocab = np_sent_level_vocab_set     # sub_vocab, slice W0
    # part_target_vocab(ndarray) is same with part_trg_vocab_nid2oid(dict) except the type
    return part_trg_vocab_nid2oid, np_sent_level_vocab_set

    '''
    if not os.path.exists(config['trg_cands_dict']):
        logger.info('start load target candidates for each source unique word by alignment')
        trg_cands_dict = trg_cands_from_aln(config['src_data'], config['trg_data'], config[
                                            'aln_data'], src_vocab, trg_vocab)
        import cPickle
        # can not use the name preprocess.dict as the python file name
        with open(config['trg_cands_pkl'], 'wb') as f:
            cPickle.dump(trg_cands_dict, f, protocol=cPickle.HIGHEST_PROTOCOL)
            logger.info('save {}'.format(config['trg_cands_pkl']))

        f = open(config['trg_cands_dict'], 'w')
        for k, v in trg_cands_dict.iteritems():
            f.writelines('{}:{}'.format(k, str(v)))  # there may be '' in the keys, because data noise
            f.write('\n')
        f.close()
        logger.info('save {}'.format(config['trg_cands_dict']))
        logger.info('done')
    else:
        logger.info('exists, load from {}'.format(config['trg_cands_pkl']))
        trg_cands_dict = pickle.load(open(config['trg_cands_pkl']))
    '''

    '''
            # obtain the target vocabulary of this batch here
            x, y = tr_data[0], tr_data[2]   # batch_size*sent_len
            lsrc_words_in_batch, lref_words_in_batch = _2dnumpy2list(x), _2dnumpy2list(y)
            laln_ref_words_in_batch = []
            for src_word_id in lsrc_words_in_batch:
                if src_word_id in trg_cands_dict:
                    laln_ref_words_in_batch = list(
                        set(laln_ref_words_in_batch) | set(trg_cands_dict[src_word_id]))
            ref_vocab = list(set(laln_ref_words_in_batch) | set(
                ltopk_trg_vocab_idx) | set(lref_words_in_batch))
    '''
