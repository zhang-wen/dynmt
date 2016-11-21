from __future__ import division

from utils import exeTime, back_tracking, init_beam, part_sort, sigmoid_better
import logging
import logging.config
logger = logging.getLogger('nbs')
# logger.setLevel(logging.DEBUG)
logger.setLevel(logging.INFO)
import numpy
import time
import sys


def beam_search_trans(src_sent, n, switchs, trg_vocab_i2w, k=10, maxlen=40, ptv=None):
    ifvalid, ifbatch, ifscore, ifnorm, ifmv, _ = switchs
    if ifvalid:
        src = src_sent[0]   # numpy ndarray
        ptv = numpy.unique(numpy.array(sorted(src_sent[1])).astype('int64'))
    else:
        src = src_sent
    # subdict set [0,2,6,29999, 333]

#<type 'list'>
#[10811, 140, 217, 19, 1047, 482, 29999]
    np_src_sent = numpy.asarray(src, dtype='int64')
#<type 'numpy.ndarray'> (7,)
#[10811   140   217    19  1047   482 29999]
    if np_src_sent.ndim == 1:  # x (5,)
        # x(5, 1), (slen, batch_size)
        np_src_sent = np_src_sent[:, None]

    maxlen = 2 * np_src_sent.shape[0]
    eos_id = len(trg_vocab_i2w) - 1
    bos_id = 0
    counter = [0, 0, 0, 0, 0, 0, 0, 0]
    if ifbatch:
        best_trans, loss = beam_search_comb(
            np_src_sent, n, switchs, maxlen, eos_id, counter, k, ptv)
    else:
        best_trans, loss = beam_search(
            np_src_sent, n, switchs, maxlen, eos_id, counter, k, ptv)
    logger.info('@source length[{}], translation length(without eos)[{}], maxlen[{}], loss'
                '[{}]'.format(len(src), len(best_trans), maxlen, loss))
    logger.info(
        'init[{}] nh[{}] na[{}] ns[{}] mo[{}] ws[{}] ps[{}] p[{}]'.format(*counter))
    return best_trans


##################################################################

# Wen Zhang: beam search, no batch

##################################################################


@exeTime
def beam_search(np_src_sent, n, switchs, maxlen, eos_id, counter, k=10, ptv=None):
    ifvalid, ifbatch, ifscore, ifnorm, ifmv, _ = switchs
    avg_loc, total_loc = 0, 0
    samples, samples_len = [], []

    context = n.encode(numpy.transpose(np_src_sent))   # np_src_sent (sl, 1), beam==1
    s_tm1 = n.dec_2_h0

    counter[0] += 1
    # (1, trg_nhids), (src_len, 1, src_nhids*2)
    beam = []
    init_beam(beam, cnt=maxlen, init_state=s_tm1)
    for i in range(1, maxlen + 1):
        if (i - 1) % 10 == 0:
            sys.stdout.write(str(i - 1))
            sys.stdout.flush()
        cands = []
        for j in xrange(len(beam[i - 1])):  # size of last beam
            if (i - 1) % 10 == 0 and j % 1000 == 0:
                sys.stdout.write('*')
                sys.stdout.flush()
            # (45.32, (beam, trg_nhids), -1, 0)
            sum_loss_by_lbeam, s_tm1, y_tm1, bp_tm1 = beam[i - 1][j]
            y_tm1_emb, h_t = n.first_hidden(y_tm1, s_tm1)
            counter[1] += 1
            p_t, a_t = n.attention(context, h_t)
            counter[2] += 1
            s_t = n.next_state(a_t, h_t)
            counter[3] += 1
            combout = n.comb_out(y_tm1_emb, s_t, a_t)
            counter[4] += 1
            logger.debug('f_ws ............... in beam search')
            s = time.time()
            next_scores = n.scores(combout, ptv)
            e = time.time()
            logger.debug(e - s)
            counter[5] += 1
            if ifscore:
                next_scores_flat = next_scores.npvalue().flatten()    # (1,vocsize) -> (vocsize,)
                ranks_idx_flat = part_sort(-next_scores_flat, k - len(samples))
                k_avg_loss_flat = -next_scores_flat[ranks_idx_flat]  # -log_p_y_given_x
            else:
                logger.debug('f_p ............... in beam search')
                s = time.time()
                next_probs = n.softmax(next_scores)  # softmax, the larger the better
                counter[7] += 1
                e = time.time()
                logger.debug(e - s)
                next_probs_flat = next_probs.npvalue().flatten()    # (1,vocsize) -> (vocsize,)
                ranks_idx_flat = part_sort(next_probs_flat, k - len(samples))
                k_avg_loss_flat = next_probs_flat[ranks_idx_flat]
            cands += [(sum_loss_by_lbeam + k_avg_loss_flat[idx], s_t, [wid], j) for idx, wid in
                      enumerate(ranks_idx_flat)]

        k_ranks_flat = part_sort(numpy.asarray([cand[0]
                                                for cand in cands] + [numpy.inf]), k - len(samples))
        k_sorted_cands = [cands[r] for r in k_ranks_flat]
        for b in k_sorted_cands:
            if b[2] == eos_id:
                logger.info('add: {}'.format(((b[0] / i), b[0]) + b[2:] + (i,)))
                if ifnorm:
                    samples.append(((b[0] / i), b[0]) + b[2:] + (i,))
                else:
                    samples.append((b[0], ) + b[2:] + (i, ))
                if len(samples) == k:
                    # output sentence, early stop, best one in k
                    logger.info('early stop! see {} samples ending with eos.'.format(k))
                    logger.info('average location of back pointers [{}/{}={}]'.format(
                        avg_loc, total_loc, format(avg_loc / total_loc, '0.3f')))
                    # if ifscore:
                    #    best_sample = sorted(samples, key=lambda tup: tup[0], reverse=True)[0]
                    # else:
                    best_sample = sorted(samples, key=lambda tup: tup[0])[0]
                    logger.info(
                        'translation length(with eos) [{}]'.format(best_sample[-1]))
                    for sample in samples:  # tuples
                        logger.info(sample)
                    return back_tracking(beam, best_sample)
            else:
                # should calculate when generate item in current beam
                avg_loc += (b[3] + 1)
                total_loc += 1
                beam[i].append(b)
        logger.debug('beam {} ----------------------------'.format(i))
        for b in beam[i]:
            logger.debug(b[0:1] + b[2:])    # do not output state

    # no early stop, back tracking
    logger.info('average location of back pointers [{}/{}={}]'.format(
        avg_loc, total_loc, format(avg_loc / total_loc, '0.3f')))
    if len(samples) == 0:
        logger.info('no early stop, no candidates ends with eos, selecting from '
                    'len {} candidates, may not end with eos.'.format(maxlen))
        best_sample = (beam[maxlen][0][0],) + beam[maxlen][0][2:] + (maxlen, )
        logger.info('translation length(with eos) [{}]'.format(best_sample[-1]))
        return back_tracking(beam, best_sample)
    else:
        logger.info('no early stop, not enough {} candidates end with eos, selecting '
                    'the best sample ending with eos from {} samples.'.format(k, len(samples)))
        # if ifscore:
        #    best_sample = sorted(samples, key=lambda tup: tup[0], reverse=True)[0]
        # else:
        best_sample = sorted(samples, key=lambda tup: tup[0])[0]
        logger.info('translation length(with eos) [{}]'.format(best_sample[-1]))
        return back_tracking(beam, best_sample)


@exeTime
def beam_search_comb(np_src_sent, fs, switchs, maxlen, eos_id, counter, k=10, ptv=None):
    f_init, f_nh, f_na, f_ns, f_mo, f_ws, f_ps, f_p = fs
    ifvalid, ifbatch, ifscore, ifnorm, ifmv, _ = switchs
    avg_loc, total_loc = 0, 0
    samples, samples_len = [], []
    hyp_scores = numpy.zeros(1).astype('float32')
    s_im1, ctx0 = f_init(np_src_sent)   # np_src_sent (sl, 1), beam==1
    counter[0] += 1
    beam = []
    init_beam(beam, cnt=maxlen, init_state=s_im1)
    for i in range(1, maxlen + 1):
        # beam search here
        if (i - 1) % 10 == 0:
            sys.stdout.write(str(i - 1))
            sys.stdout.flush()

        prevb = beam[i - 1]
        len_prevb = len(prevb)
        cands = []
        y_im1 = numpy.array([b[2] for b in prevb])
        # (src_sent_len, 1, 2*src_nhids)->(src_sent_len, len_prevb, 2*src_nhids)
        ctx = numpy.tile(ctx0, [len_prevb, 1])
        # print len_prevb, k - len(samples)
        # ctx = numpy.tile(ctx0, [k - len(samples), 1]), for beam 1, batch_size is 1
        hi = f_nh(*[y_im1, s_im1])
        counter[1] += 1
        pi, ai = f_na(*[ctx, hi])
        counter[2] += 1
        s_im1 = f_ns(*[hi, ai])
        counter[3] += 1
        mo = f_mo(*[y_im1, ai, s_im1])
        counter[4] += 1
        next_scores = f_ws(*[mo, ptv])  # the larger the better
        counter[5] += 1
        if ifscore:
            if False:
                next_scores_sigmoid = sigmoid_better(next_scores)
                next_loss = -numpy.log(next_scores_sigmoid)
            else:
                next_loss = -next_scores  # -log_p_y_given_x
        else:
            next_loss = f_p(next_scores)  # (denomi sum, softmax)
            counter[7] += 1
        cand_scores = hyp_scores[:, None] + next_loss
        cand_scores_flat = cand_scores.flatten()
        ranks_flat = part_sort(cand_scores_flat, k - len(samples))  # problem here, not k
        voc_size = next_scores.shape[1]
        prevb_id = ranks_flat // voc_size
        word_indices = ranks_flat % voc_size
        costs = cand_scores_flat[ranks_flat]

        for b in zip(costs, s_im1[prevb_id], word_indices, prevb_id):
            if b[2] == eos_id:
                if ifnorm:
                    samples.append(((b[0] / i), b[0]) + b[2:] + (i, ))
                else:
                    samples.append((b[0], ) + b[2:] + (i,))
                if len(samples) == k:
                    # output sentence, early stop, best one in k
                    logger.info('early stop! see {} samples ending with eos.'.format(k))
                    logger.info('average location of back pointers [{}/{}={}]'.format(
                        avg_loc, total_loc, format(avg_loc / total_loc, '0.3f')))
                    best_sample = sorted(samples, key=lambda tup: tup[0])[0]
                    logger.info(
                        'translation length(with eos) [{}]'.format(best_sample[-1]))
                    for sample in samples:  # tuples
                        logger.info(sample)
                    return back_tracking(beam, best_sample)
            else:
                # should calculate when generate item in current beam
                avg_loc += (b[3] + 1)
                total_loc += 1
                beam[i].append(b)
        logger.debug('beam {} ----------------------------'.format(i))
        for b in beam[i]:
            logger.debug(b[0:1] + b[2:])    # do not output state
        hyp_scores = numpy.array([b[0] for b in beam[i]])
        # batch state of current beam
        s_im1 = numpy.array([b[1] for b in beam[i]])

    # no early stop, back tracking
    logger.info('average location of back pointers [{}/{}={}]'.format(
        avg_loc, total_loc, format(avg_loc / total_loc, '0.3f')))
    if len(samples) == 0:
        logger.info('no early stop, no candidates ends with eos, selecting from '
                    'len{} candidates, may not end with eos.'.format(maxlen))
        best_sample = (beam[maxlen][0][0],) + beam[maxlen][0][2:] + (maxlen, )
        logger.info('translation length(with eos) [{}]'.format(best_sample[-1]))
        return back_tracking(beam, best_sample)
    else:
        logger.info('no early stop, not enough {} candidates end with eos, selecting '
                    'the best sample ending with eos from {} samples.'.format(k, len(samples)))
        best_sample = sorted(samples, key=lambda tup: tup[0])[0]
        logger.info('translation length(with eos) [{}]'.format(best_sample[-1]))
        return back_tracking(beam, best_sample)
