from __future__ import division

from utils import exeTime, _filter_reidx, part_sort
import numpy
import logging
import logging.config
logger = logging.getLogger('naive_beam_search')
logger.setLevel(logging.DEBUG)
import copy


@exeTime
def original_trans(x, fs, switchs, trg_vocab_i2w, k=10, maxlen=40):
    counter = [0, 0, 0, 0, 0, 0, 0, 0]
    f_init, f_nh, f_na, f_ns, f_mo, f_ws, f_ps, f_p = fs
    ifvalid, ifbatch, ifscore, ifnorm, ifmv = switchs

    x = x[0] if ifvalid else x  # numpy ndarray
    # subdict set [0,2,6,29999, 333]
    sub_trg_vocab_i2w = numpy.asarray(x[1], dtype='int32') if ifvalid else None

    # k is the beam size we have
    x = numpy.asarray(x, dtype='int64')
    if x.ndim == 1:
        x = x[None, :]
    src_sent_len = x.shape[1]
    maxlen = src_sent_len * 2
    x = x.T
    eos_id = len(trg_vocab_i2w) - 1
    bos_id = 0

    sample = []
    sample_score = []

    live_k = 1
    dead_k = 0

    hyp_samples = [[]] * live_k
    hyp_scores = numpy.zeros(live_k).astype('float32')
    hyp_states = []

    # get initial state of decoder rnn and encoder context
    s_im1, ctx0 = f_init(x)
    counter[0] += 1
    y_im1 = [-1]  # indicator for the first target word (bos target)

    for ii in xrange(maxlen):
        # (src_sent_len, 1, 2*src_nhids) -> (src_sent_len, live_k, 2*src_nhids)
        ctx = numpy.tile(ctx0, [live_k, 1])
        hi = f_nh(*[y_im1, s_im1])
        counter[1] += 1
        pi, ai = f_na(*[ctx, hi])
        counter[2] += 1
        s_im1 = f_ns(*[hi, ai])  # note, s_im1 should be updated!
        counter[3] += 1
        mo = f_mo(*[y_im1, ai, s_im1])
        counter[4] += 1
        next_scores = f_ws(*[mo])  # the larger the better
        counter[5] += 1
        if ifscore:
            if False:
                next_scores_sigmoid = sigmoid_better(next_scores)
                next_loss = -numpy.log(next_scores_sigmoid)
            else:
                next_loss = -next_scores
        else:
            next_loss = f_p(next_scores)   # the larger the better
            counter[7] += 1
            #next_loss = -numpy.log(next_probs)
        #cand_scores = hyp_scores[:, None] - numpy.log(next_scores)
        cand_scores = hyp_scores[:, None] + next_loss
        # print ii, ' ==============================================='
        # print next_scores
        # print ii, ' ==============================================='
        # print cand_scores
        cand_flat = cand_scores.flatten()
        # ranks_flat = cand_flat.argsort()[:(k-dead_k)]
        # we do not need to generate k candidate here, because we just need to generate k-dead_k
        # more candidates ending with eos, so for each previous candidate we just need to expand
        # k-dead_k candidates
        ranks_flat = part_sort(cand_flat, k - dead_k)
        # print ranks_flat, cand_flat[ranks_flat[1]], cand_flat[ranks_flat[8]]

        voc_size = next_scores.shape[1]
        trans_indices = ranks_flat // voc_size
        word_indices = ranks_flat % voc_size
        costs = cand_flat[ranks_flat]

        new_hyp_samples = []
        new_hyp_scores = numpy.zeros(k - dead_k).astype('float32')
        new_hyp_states = []

        for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
            new_hyp_samples.append(hyp_samples[ti] + [wi])
            new_hyp_scores[idx] = copy.copy(costs[idx])
            new_hyp_states.append(copy.copy(s_im1[ti]))

        # check the finished samples
        new_live_k = 0
        hyp_samples = []
        hyp_scores = []
        hyp_states = []
        # current beam, if the hyposise ends with eos, we do not
        for idx in xrange(len(new_hyp_samples)):
            if new_hyp_samples[idx][-1] == eos_id:
                sample.append(new_hyp_samples[idx])
                sample_score.append(new_hyp_scores[idx])
                # print new_hyp_scores[idx], new_hyp_samples[idx]
                dead_k += 1
            else:
                new_live_k += 1
                hyp_samples.append(new_hyp_samples[idx])
                hyp_scores.append(new_hyp_scores[idx])
                hyp_states.append(new_hyp_states[idx])
        hyp_scores = numpy.array(hyp_scores)
        live_k = new_live_k
        # print ii, '====================================================='
        # print 'hyp_scores:'
        # print hyp_scores
        # print 'hyp_samples:'
        # for hyp_sample in hyp_samples:
        #    print hyp_sample

        if new_live_k < 1:
            break
        if dead_k >= k:
            break

        y_im1 = numpy.array([w[-1] for w in hyp_samples])
        s_im1 = numpy.array(hyp_states)

    if live_k > 0:
        for idx in xrange(live_k):
            sample.append(hyp_samples[idx])
            sample_score.append(hyp_scores[idx])

    if ifnorm:
        lengths = numpy.array([len(s) for s in sample])
        avg_sample_score = sample_score / lengths
    else:
        avg_sample_score = sample_score
    sidx = numpy.argmin(avg_sample_score)

    best_sum_loss = sample_score[sidx]
    best_avg_loss = avg_sample_score[sidx]
    best_trans = sample[sidx]

    logger.info('@source length[{}], translation length(with eos)[{}], maxlen[{}], avg loss'
                '[{}]={}/{}'.format(src_sent_len, len(best_trans), maxlen, avg_sample_score[sidx],
                                    sample_score[sidx], lengths[sidx]))
    logger.info('init[{}] nh[{}] na[{}] ns[{}] mo[{}] ws[{}] ps[{}] p[{}]'.format(*counter))
    return _filter_reidx(bos_id, eos_id, best_trans, trg_vocab_i2w, ifmv, sub_trg_vocab_i2w)
