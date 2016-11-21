# -*- coding: utf-8 -*-

from __future__ import division
import numpy
from itertools import izip
import time
import sys

# exeTime


def exeTime(func):
    def newFunc(*args, **args2):
        t0 = time.time()
        sys.stderr.write('@{}, {} start\n'.format(time.strftime(
            "%X", time.localtime()), func.__name__))
        # print "@%s, {%s} start" % (time.strftime("%X", time.localtime()),
        # func.__name__)
        back = func(*args, **args2)
        sys.stderr.write('@{}, {} end\n'.format(time.strftime(
            "%X", time.localtime()), func.__name__))
        # print "@%s, {%s} end" % (time.strftime("%X", time.localtime()),
        # func.__name__)
        sys.stderr.write('@{}s taken for {}\n'.format(
            format(time.time() - t0, '0.3f'), func.__name__))
        # print "@%.3fs taken for {%s}" % (time.time() - t0, func.__name__)
        return back
    return newFunc

import os
import shutil


def init_dirs(models_dir, valid_dir, test_dir, **kwargs):
    if os.path.exists(models_dir):
        shutil.rmtree(models_dir)
        sys.stderr.write('{} exists, delete\t'.format(models_dir))
    os.mkdir(models_dir)
    sys.stderr.write('create {}\n'.format(models_dir))

    if not valid_dir == '':
        if os.path.exists(valid_dir):
            shutil.rmtree(valid_dir)
            sys.stderr.write('{} exists, delete\t'.format(valid_dir))
        os.mkdir(valid_dir)
        sys.stderr.write('create {}\n'.format(valid_dir))

    if not test_dir == '':
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
            sys.stderr.write('{} exists, delete\t'.format(test_dir))
        os.mkdir(test_dir)
        sys.stderr.write('create {}\n'.format(test_dir))


def _index2sentence(vec, dic, part=None):
    if part is not None:
        # OrderedDict([(0, 0), (1, 1), (3, 5), (8, 2), (10, 3), (100, 4)])
        # reverse: OrderedDict([(0, 0), (1, 1), (5, 3), (2, 8), (3, 10), (4, 100)])
        # part[index] get the real index in large target vocab firstly
        r = [dic[part[index]] for index in vec]
    else:
        r = [dic[index] for index in vec]
    return " ".join(r)


def _filter_reidx(bos_id, eos_id, best_trans, ifmv=False, tv=None):
    if ifmv:
        best_trans = filter(lambda y: tv[y] != eos_id and tv[y] != bos_id, best_trans)
    else:
        best_trans = filter(lambda y: y != eos_id and y != bos_id, best_trans)
    return best_trans


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


def part_sort_lg2sm(vec, klargest):
    # numpy.argpartition(vec, num)
    sz = len(vec)
    left = sz - klargest
    ind = numpy.argpartition(vec, left)[left:]
    # sort the small data before the num_th element, and get the index
    # index of k largest elements (lg to sm)
    return ind[numpy.argsort(-vec[ind])]


def euclidean(n1, n2):
    # numpy.float64 (0. ~ inf)
    return numpy.linalg.norm(n1 - n2)


def sigmoid(x):
    return 1 / (1 + numpy.exp(-x))


def sigmoid_better(x):
    return (numpy.arctan(x / 100) / numpy.pi) + 0.5


def logistic(x):
    return 1 / (1 + numpy.exp(-x / 10000.))


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = numpy.exp(x - numpy.max(x))
    return e_x / e_x.sum()


def kl_dist(p, q):
    p = numpy.asarray(p, dtype=numpy.float)
    q = numpy.asarray(q, dtype=numpy.float)
    return numpy.sum(numpy.where(p != 0, (p - q) * numpy.log10(p / q), 0))


def back_tracking(beam, best_sample_endswith_eos):
    # (0.76025655120611191, [29999], 0, 7)
    if len(best_sample_endswith_eos) == 5:
        best_loss, accum, w, bp, endi = best_sample_endswith_eos
    else:
        best_loss, w, bp, endi = best_sample_endswith_eos
    # from previous beam of eos beam, firstly bp:j is the item index of
    # {end-1}_{th} beam
    seq = []
    for i in reversed(xrange(1, endi)):
        # the best (minimal sum) loss which is the first one in the last beam,
        # then use the back pointer to find the best path backward
        # contain eos last word, finally we filter, so no matter
        _, _, w, backptr = beam[i][bp]
        seq.append(w[0])
        bp = backptr
    return seq[::-1], best_loss  # reverse


def init_beam(beam, cnt=50, init_score=0.0, init_state=None):
    for i in range(cnt + 1):
        ibeam = []  # one beam [] for one char besides start beam
        beam.append(ibeam)
    # (sum score, state, backptr), indicator for the first target word (bos <S>)
    beam[0].append((init_score, init_state, [0], 0))
    # such as: beam[0] is (0.0 (-log(1)), 'Could', 0)


def dec_conf(switchs, k, mode, kl, nprocess, lm, ngram):
    ifvalid, ifbatch, ifscore, ifnorm, ifmv, merge_way = switchs
    sys.stderr.write(
        '\n.........................decoder config..........................\n')
    if mode == 0:
        sys.stderr.write('# MLE search => ')
    elif mode == 1:
        sys.stderr.write('# Original beam search => ')
    elif mode == 2:
        sys.stderr.write('# Naive beam search => ')
    elif mode == 3:
        sys.stderr.write('# Cube pruning => ')
    sys.stderr.write('\n\t beam size: {}'
                     '\n\t kl_threshold: {}'
                     '\n\t n_process: {}'
                     '\n\t decode valid: {}'
                     '\n\t use batch: {}'
                     '\n\t use softmax: {}'
                     '\n\t use normalized: {}'
                     '\n\t manipulate vocab: {}'
                     '\n\t cube pruning merge way: {}'
                     '\n\t language model: {}'
                     '\n\t ngrams restored into trie: {}\n\n'.format(
                         k,
                         kl,
                         nprocess,
                         True if ifvalid else False,
                         True if ifbatch else False,
                         False if ifscore else True,
                         True if ifnorm else False,
                         True if ifmv else False,
                         merge_way,
                         lm,
                         ngram)
                     )
    sys.stdout.flush()
