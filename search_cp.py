from __future__ import division

import time
import sys
import numpy

from utils import _filter_reidx, part_sort, exeTime, kl_dist, \
    sigmoid, sigmoid_better, euclidean, softmax, init_beam, back_tracking
from collections import OrderedDict
import heapq
from itertools import count
import copy

import logging
import logging.config
logger = logging.getLogger('cp')
logger.setLevel(logging.DEBUG)
# logger.setLevel(logging.INFO)
from wlm import vocab_prob_given_ngram

##################################################################

# NOTE: merge all candidates in previous beam by euclidean distance of two state vector or KL
# distance of alignment probability

##################################################################


def merge(prevb, fs, switchs, ctx0, eq_classes, k, lqc, kl):
    merge_way = switchs[-1]
    len_prevb = len(prevb)
    used = []
    key = 0

    _memory = [None] * len_prevb
    for j in range(len_prevb):  # index of each item in last beam
        if j in used:
            continue

        tmp = []
        if _memory[j]:
            _needed = _memory[j]
            if merge_way == 'Him1':
                _, y_im1_1, s_im1_1, _, nj, _ = _needed
            elif merge_way == 'Hi':
                _, y_im1_1, hi_1, _, nj, _ = _needed
            elif merge_way == 'AiKL':
                _, y_im1_1, hi_1, ai_1, nj, pi_1 = _needed
            else:
                sys.stderr.out('Unrecogined type', merge_way)
            assert(j == nj)
        else:
            # calculation
            score_im1_1, s_im1_1, y_im1_1, bp_im1_1 = prevb[j]
            if merge_way == 'Him1':
                _needed = _memory[j] = (score_im1_1, y_im1_1, s_im1_1, None, j, None)
            else:
                hi_1 = fs[1](*[[y_im1_1], s_im1_1])
                lqc[1] += 1
                if merge_way == 'Hi':
                    _needed = _memory[j] = (score_im1_1, y_im1_1, hi_1, None, j, None)
                elif merge_way == 'AiKL':
                    pi_1, ai_1 = fs[2](*[ctx0, hi_1])
                    lqc[2] += 1
                    _needed = _memory[j] = (
                        score_im1_1, y_im1_1, hi_1, ai_1, j, pi_1)
                else:
                    sys.stderr.out('Unrecogined type', merge_way)

        tmp.append(_needed[:-1])

        for jj in range(j + 1, len_prevb):
            if _memory[jj]:
                _needed = _memory[jj]
                if merge_way == 'Him1':
                    _, y_im1_2, s_im1_2, _, njj, _ = _needed
                elif merge_way == 'Hi':
                    _, y_im1_2, hi_2, _, njj, _ = _needed
                elif merge_way == 'AiKL':
                    _, y_im1_2, hi_2, ai_2, njj, pi_2 = _needed
                else:
                    sys.stderr.out('Unrecogined type', merge_way)
                assert(jj == njj)
            else:   # calculation
                score_im1_2, s_im1_2, y_im1_2, bp_im1_2 = prevb[jj]
                if merge_way == 'Him1':
                    _needed = _memory[jj] = (score_im1_2, y_im1_2, s_im1_2, None, jj, None)
                else:
                    hi_2 = fs[1](*[[y_im1_2], s_im1_2])
                    lqc[1] += 1
                    if merge_way == 'Hi':
                        _needed = _memory[jj] = (
                            score_im1_2, y_im1_2, hi_2, None, jj, None)
                    elif merge_way == 'AiKL':
                        pi_2, ai_2 = fs[2](*[ctx0, hi_2])
                        lqc[2] += 1
                        _needed = _memory[jj] = (
                            score_im1_2, y_im1_2, hi_2, ai_2, jj, pi_2)
                    else:
                        sys.stderr.out('Unrecogined type', merge_way)

            if merge_way == 'Him1':
                logger.debug('{} {} {}, {}'.format(
                    y_im1_1, y_im1_2, euclidean(s_im1_2, s_im1_1), kl))
                ifmerge = (y_im1_2 == y_im1_1 and euclidean(s_im1_2, s_im1_1) < kl)
            elif merge_way == 'Hi':
                ifmerge = (y_im1_2 == y_im1_1 and euclidean(hi_2, hi_1) < kl)
            elif merge_way == 'AiKL':
                dist = kl_dist(pi_2, pi_1)
                logger.debug('kl distance: {}'.format(dist))
                ifmerge = (y_im1_2 == y_im1_1 and dist < kl)
            if ifmerge:
                tmp.append(_needed[:-1])
                used.append(jj)

        eq_classes[key] = tmp
        key += 1
    # print ',,,,,,', lqc[1]


##################################################################

# NOTE: (Wen Zhang) create cube by sort row dimension

##################################################################


#@exeTime
def create_cube(sample_len, fs, switchs, context, k, eq_classes, lqc, lm, tvcb, tvcb_i2w):
    # eq_classes: (score_im1, y_im1, hi, ai, loc_in_prevb) NEW
    f_init, f_nh, f_na, f_ns, f_mo, f_ws, f_ps, f_p = fs
    ifvalid, ifbatch, ifscore, ifnorm, ifmv, merge_way = switchs
    cube = []
    cnt = count()
    for idx, leq_class in eq_classes.iteritems():   # sub cube
        _0score_im1, _0y_im1, _sim1_or_hi, _ai, _0j = leq_class[0]
        subcube = []

        if lm is not None and not _0y_im1 == -1:
            # TODO sort the row dimension by the distribution of next words based on language model
            print '[', _0y_im1, ']'
            logps, words = vocab_prob_given_ngram(lm, [_0y_im1], tvcb, tvcb_i2w)
            _neg_logp_ith_flat = -numpy.asarray(logps)
            _k_rank_idx = part_sort(_neg_logp_ith_flat, k - sample_len)
            _k_ith_neg_log_prob = _neg_logp_ith_flat[_k_rank_idx]
            print _k_ith_neg_log_prob
            for idx in _k_rank_idx:
                print words[idx],
            print

            _0avg_hi = _0avg_ai = None
            avg_attention = True
            if avg_attention:
                if not len(leq_class) == 1:
                    if merge_way == 'Him1':
                        merged_s_im1 = [tup[2] for tup in leq_class]
                        _s_im1 = numpy.mean(numpy.array(merged_s_im1), axis=0)
                        _0avg_hi = f_nh(*[[_0y_im1], _s_im1])
                        lqc[1] += 1
                        _, _0avg_ai = f_na(*[context, _0avg_hi])
                        lqc[2] += 1
                else:
                    _0avg_hi = f_nh(*[[_0y_im1], leq_class[0][2]])
                    lqc[1] += 1
                    _, _0avg_ai = f_na(*[context, _0avg_hi])
                    lqc[2] += 1

            for i, tup in enumerate(leq_class):
                subcube.append([((float(tup[0] + _k_ith_neg_log_prob[j]), next(cnt),
                                  _k_ith_neg_log_prob[j]) + tup + (_0avg_hi, _0avg_ai, wid, i, j))
                                for j, wid in enumerate(_k_rank_idx)])
        else:
            # TODO sort the row dimension by average scores
            if not len(leq_class) == 1:
                if merge_way == 'Him1':
                    merged_s_im1 = [tup[2] for tup in leq_class]
                    _sim1_or_hi = numpy.mean(numpy.array(merged_s_im1), axis=0)
                elif merge_way == 'Hi':
                    merged_hi = [tup[2] for tup in leq_class]
                    _sim1_or_hi = numpy.mean(numpy.array(merged_hi), axis=0)
                elif merge_way == 'AiKL':
                    merged_h = [tup[2] for tup in leq_class]
                    merged_a = [tup[3] for tup in leq_class]
                    _sim1_or_hi = numpy.mean(numpy.array(merged_h), axis=0)
                    _ai = numpy.mean(numpy.array(merged_a), axis=0)

            if merge_way == 'Him1':
                _0avg_hi = f_nh(*[[_0y_im1], _sim1_or_hi])
                lqc[1] += 1
                _, _0avg_ai = f_na(*[context, _0avg_hi])
                lqc[2] += 1
                _si = f_ns(*[_0avg_hi, _0avg_ai])
                lqc[3] += 1
                _mo = f_mo(*[[_0y_im1], _0avg_ai, _si])
                lqc[4] += 1
                _scores_ith = f_ws(_mo)  # the larger the better
                lqc[5] += 1
            elif merge_way == 'Hi':
                _, _1avg_ai = f_na(*[context, _sim1_or_hi])
                lqc[2] += 1
                _si = f_ns(*[_sim1_or_hi, _1avg_ai])
                lqc[3] += 1
                _mo = f_mo(*[[_0y_im1], _1avg_ai, _si])
                lqc[4] += 1
                _scores_ith = f_ws(_mo)  # the larger the better
                lqc[5] += 1
            elif merge_way == 'AiKL':
                _si = f_ns(*[_sim1_or_hi, _ai])
                lqc[3] += 1
                _mo = f_mo(*[[_0y_im1], _ai, _si])
                lqc[4] += 1
                _scores_ith = f_ws(_mo)  # the larger the better
                lqc[5] += 1

            logger.debug(
                'sort row dimension by score, if score is the same, softmax prob must be the same')
            if ifscore:
                _scores_ith_flat = _scores_ith.flatten()
                _k_rank_idx = part_sort(-_scores_ith_flat, k - sample_len)
                _k_scores_ith_flat = _scores_ith_flat[_k_rank_idx]
                _k_ith_neg_log_prob = -_k_scores_ith_flat
            else:
                _probs_ith = f_p(_scores_ith)
                lqc[7] += 1
                _probs_ith_flat = _probs_ith.flatten()
                _k_rank_idx = part_sort(_probs_ith_flat, k - sample_len)
                _k_ith_neg_log_prob = _probs_ith_flat[_k_rank_idx]

            # add cnt for error The truth value of an array with more than one
            # element is ambiguous
            for i, tup in enumerate(leq_class):
                subcube.append([((float(tup[0] + _k_ith_neg_log_prob[j]), next(cnt),
                                  _k_ith_neg_log_prob[j]) + tup + (_si, None, wid, i, j))
                                for j, wid in enumerate(_k_rank_idx)])
        # sort in previous loss ascending order (by column)
        # cube.append(sorted(subcube, key=lambda tup: tup[0]))
        cube.append(subcube)
    '''
    logger.debug(
        'cube:%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')   # sorted row
    ngroup = len(cube)
    for groupid in xrange(ngroup):
        group = cube[groupid]
        nmergings = len(group)
        logger.debug('group: {} contains {} mergings:'.format(
            groupid, nmergings))
        for mergeid in xrange(nmergings):
            for cubetup in group[mergeid]:
                print '{}({}+{})'.format(format(cubetup[0], '0.3f'), format(cubetup[-4], '0.3f'),
                                         format(cubetup[2], '0.3f')),
                # print '{}({}+{})'.format(cubetup[0], cubetup[-4], cubetup[2]),
            print '\n'
    logger.debug(
        '%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')   # sorted row
    '''
    return cube

##################################################################

# NOTE: (Wen Zhang) cube pruning

##################################################################


def cube_prune(bi, samples, whichbeam, fs, switchs, context, k, eos_id, cube, locrt, lqc, lm=None):
    f_init, f_nh, f_na, f_ns, f_mo, f_ws, f_ps, f_p = fs
    ifvalid, ifbatch, ifscore, ifnorm, ifmv, merge_way = switchs
    # search in cube
    nsubcube = len(cube)    # cube (matrix(mergings) or vector(no mergings))
    each_subcube_colsz, each_subcube_rowsz = [], []
    cube_size, counter = 0, 0
    extheap, wavetag, buf_state_merge = [], [], []
    lqc[8] += nsubcube   # count of total sub-cubes
    for whichsubcube in xrange(nsubcube):
        subcube = cube[whichsubcube]
        rowsz = len(subcube)
        each_subcube_rowsz.append(rowsz)
        each_subcube_colsz.append(len(subcube[0]))
        # print whichbeam, rowsz
        lqc[9] += rowsz   # count of total lines in sub-cubes
        # initial heap, starting from the left-top corner (best word) of each subcube
        # real score here ... may adding language model here ...
        heapq.heappush(extheap, subcube[0][0] + (whichsubcube,))
        buf_state_merge.append([])

    sample_len = len(samples)
    maxk = k - sample_len
    while extheap and counter < maxk:
        _score_i, _, _, score_im1, y_im1, sim1_or_hi, ai, prevb_idx, _hi_or_si, _ai, y_i, \
            iexpend, jexpend, which = heapq.heappop(extheap)
        if lm is not None and not y_im1 == -1:
            if _hi_or_si is not None and _ai is not None:
                si = f_ns(*[_hi_or_si, _ai])
                lqc[3] += 1
                moi = f_mo(*[[y_im1], _ai, si])
                lqc[4] += 1
                score_ith = f_ps(*[moi, y_i])
                lqc[6] += 1
                score_i = score_im1 - score_ith.flatten()[0]
            true_si = si
            true_sci = score_i
        else:
            if each_subcube_rowsz[which] == 1:
                true_sci = _score_i
                true_si = _hi_or_si
            else:
                # get real distribution of the this line
                if jexpend == 0:  # down-pop
                    logger.debug('down...')
                    if merge_way == 'Him1':
                        h0i = f_nh(*[[y_im1], sim1_or_hi])
                        lqc[1] += 1
                        _, a0i = f_na(*[context, h0i])
                        lqc[2] += 1
                        si = f_ns(*[h0i, a0i])
                        lqc[3] += 1
                        moi = f_mo(*[[y_im1], a0i, si])
                        lqc[4] += 1
                    elif merge_way == 'Hi':
                        _, a1i = f_na(*[context, sim1_or_hi])
                        lqc[2] += 1
                        si = f_ns(*[sim1_or_hi, a1i])
                        lqc[3] += 1
                        moi = f_mo(*[[y_im1], a1i, si])
                        lqc[4] += 1
                    elif merge_way == 'AiKL':
                        si = f_ns(*[sim1_or_hi, ai])
                        lqc[3] += 1
                        moi = f_mo(*[[y_im1], ai, si])
                        lqc[4] += 1

                    if ifscore:
                        buf_state_merge[which].append((si, moi))
                        score_ith = f_ps(*[moi, y_i])
                        lqc[6] += 1
                        score_i = score_im1 - score_ith.flatten()[0]
                    else:
                        sci = f_ws(moi)
                        lqc[5] += 1
                        pi = f_p(sci)
                        lqc[7] += 1

                        buf_state_merge[which].append((si, pi))
                        score_i = score_im1 + pi.flatten()[y_i]
                        logger.debug('| {}={}+{}'.format(format(score_i, '0.3f'), format(score_im1,
                                                                                         '0.3f'),
                                                         format(pi.flatten()[y_i], '0.3f')))
                else:   # right-pop
                    logger.debug('right...')
                    if ifscore:
                        si, moi = buf_state_merge[which][iexpend]
                        psci = f_ps(*[moi, y_i])
                        lqc[6] += 1
                        score_i = score_im1 - psci.flatten()[0]
                    else:
                        si, pi = buf_state_merge[which][iexpend]
                        score_i = score_im1 + pi.flatten()[y_i]
                        logger.debug('-> {}={}+{}'.format(format(score_i, '0.3f'), format(score_im1,
                                                                                          '0.3f'),
                                                          format(pi.flatten()[y_i], '0.3f')))
                true_sci = score_i
                true_si = si

        if y_i == eos_id:
            # beam items count decrease 1
            if ifnorm:
                samples.append(((true_sci / (whichbeam + 1)), true_sci,
                                y_i, prevb_idx, whichbeam + 1))
            else:
                samples.append(true_sci, y_i, prevb_idx, whichbeam + 1)
            logger.debug('add sample {} {} {} {} {}'.format(
                samples[-1][0], samples[-1][1], samples[-1][2], samples[-1][3], samples[-1][4]))
            if len(samples) == k:
                # last beam created and finish cube pruning
                return True
        else:
            # generate one item in current beam
            locrt[0] += (prevb_idx + 1)
            locrt[1] += 1
            bi.append((true_sci, true_si, y_i, prevb_idx))

        whichsubcub = cube[which]
        # make sure we do not add repeadedly
        if jexpend + 1 < each_subcube_colsz[which]:
            right = whichsubcub[iexpend][jexpend + 1]
            heapq.heappush(extheap, (right + (which,)))
        if iexpend + 1 < each_subcube_rowsz[which]:
            down = whichsubcub[iexpend + 1][jexpend]
            heapq.heappush(extheap, (down + (which,)))
        counter += 1
        # beam[i] created, len: k-len(samples)
    return False


@exeTime
def cube_pruning(beam, fs, switchs, ctx0, maxlen, eos_id, k, lqc, kl, lm, tvcb, tvcb_i2w):
    locrt = [0, 0]
    samples = []
    for i in range(1, maxlen + 1):
        # beam search with cube pruning here
        eq_classes = OrderedDict()
        merge(beam[i - 1], fs, switchs, ctx0, eq_classes, k, lqc, kl)

        # create cube and generate next beam from cube
        cube = create_cube(len(samples), fs, switchs,
                           ctx0, k, eq_classes, lqc, lm, tvcb, tvcb_i2w)
        if cube_prune(beam[i], samples, i - 1, fs, switchs,
                      ctx0, k, eos_id, cube, locrt, lqc, lm):
            logger.info(
                'early stop! see {} samples ending with eos when merging.'.format(k))
            logger.info('average location of back pointers [{}/{}={}]'.format(
                locrt[0], locrt[1], format(locrt[0] / locrt[1], '0.3f')))
            best_sample = sorted(samples, key=lambda tup: tup[0])[0]
            logger.info('translation length(with eos) [{}]'.format(
                best_sample[-1]))
            for sample in samples:  # tuples
                logger.info(sample)
            return back_tracking(beam, best_sample)

        beam[i] = sorted(beam[i], key=lambda tup: tup[0])
        logger.debug('beam {} ----------------------------'.format(i))
        for b in beam[i]:
            logger.debug(b[0:1] + b[2:])    # do not output state
        # because of the the estimation of P(f|abcd) as P(f|cd), so the generated beam by
        # cube pruning may out of order by loss, so we need to sort it again here
        # losss from low to high
    # no early stop, back tracking
    logger.info('average location of back pointers [{}/{}={}]'.format(
        locrt[0], locrt[1], format(locrt[0] / locrt[1], '0.3f')))
    if len(samples) == 0:
        logger.info('no early stop, no candidates ends with eos, selecting from '
                    'len {} candidates, may not end with eos.'.format(maxlen))
        best_sample = (beam[maxlen][0][0],) + beam[maxlen][0][2:] + (maxlen, )
        logger.info(
            'translation length(with eos) [{}]'.format(best_sample[-1]))
        return back_tracking(beam, best_sample)
    else:
        logger.info('no early stop, not enough {} candidates end with eos, selecting '
                    'the best sample ending with eos from {} samples.'.format(k, len(samples)))
        best_sample = sorted(samples, key=lambda tup: tup[0])[0]
        logger.info(
            'translation length(with eos) [{}]'.format(best_sample[-1]))
        return back_tracking(beam, best_sample)


def cube_prune_trans(src_sent, fs, switchs, trg_vocab_i2w, k, kl, lm, tvcb):
    ifvalid, ifbatch, ifscore, ifnorm, ifmv, _ = switchs
    src_sent = src_sent[0] if ifvalid else src_sent  # numpy ndarray

    lqc = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    sub_trg_vocab_i2w = numpy.asarray(
        src_sent[1], dtype='int32') if ifvalid else None
    np_src_sent = numpy.asarray(src_sent, dtype='int64')
    if np_src_sent.ndim == 1:  # x (5,)
        # x(5, 1), (src_sent_len, batch_size)
        np_src_sent = np_src_sent[:, None]

    src_sent_len = np_src_sent.shape[0]
    maxlen = 2 * src_sent_len     # x(src_sent_len, batch_size)

    s_im1, ctx0 = fs[0](np_src_sent)   # np_src_sent (sl, 1), beam==1
    lqc[0] += 1
    # (1, trg_nhids), (src_len, 1, src_nhids*2)
    beam = []
    init_beam(beam, cnt=maxlen, init_state=s_im1)

    eos_id = len(trg_vocab_i2w) - 1
    bos_id = 0
    best_trans, best_loss = cube_pruning(
        beam, fs, switchs, ctx0, maxlen, eos_id, k, lqc, kl, lm, tvcb, tvcb_i2w=trg_vocab_i2w)

    logger.info('@source length[{}], translation length(without eos)[{}], maxlen[{}], loss'
                '[{}]'.format(src_sent_len, len(best_trans), maxlen, best_loss))
    logger.info(
        'init[{}] nh[{}] na[{}] ns[{}] mo[{}] ws[{}] ps[{}] p[{}]'.format(
            *lqc[0:8])
    )
    logger.info(
        'average merge count[{}/{}={}]'.format(lqc[9],
                                               lqc[8], format(lqc[9] / lqc[8], '0.3f'))
    )
    return _filter_reidx(bos_id, eos_id, best_trans, trg_vocab_i2w, ifmv, sub_trg_vocab_i2w)
