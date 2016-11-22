import kenlm
#from pytrie import SortedStringTrie as trie
#from pytrie import Trie as trie
from pytrie import StringTrie as trie
import sys
from stream_with_dict import get_tr_stream
import time


# this is to slow !!!  change to another way
def load_language_model(lm, config, trg_vocab, trg_vocab_i2w, ngram, ltrie):
    import kenlm
    m = kenlm.Model(lm)
    sys.stderr.write('use {}-gram langauge model\n'.format(m.order))
    ngram = m.order if ngram > m.order else ngram
    tr_stream = get_tr_stream(**config)
    ltrie = []
    sys.stderr.write('\tload target language model into prefix trie ...')
    idx = 0
    for tr_data in tr_stream.get_epoch_iterator():
        by = tr_data[2]
        idx += 1
        if idx % 5000 == 0:
            logger.debug(idx)
        for y in by:
            y_filter = filter(lambda x: x != 0 and x != config['trg_vocab_size'] - 1, y)
            v_yw = id2w(trg_vocab_i2w, y_filter)
            get_ngram_vocab_prob(m, trg_vocab.keys(), v_yw, ngram, ltrie)
    sys.stderr.write('\tdone')


def id2w(trg_vocab_i2w, y):
    yw = []
    for yy in y:
        yw.append(trg_vocab_i2w[yy])
    print 'id2w...', yw
    return yw


def w2id(trg_vocab, ltrie):
    for t in ltrie:
        for k in t.keys():
            v_ws = k.split()
            v_wids = []
            for w in v_ws:
                v_wids.append(str(trg_vocab[w]))
            vv = t.pop(k)
            new_vv = []
            for v in vv:
                new_vv.append(v[:-1] + (trg_vocab[v[-1]], ))
            t[' '.join(v_wids)] = new_vv

'''
##################################################################

NOTE: (Wen Zhang) get the probability distribution of all words in the vocabulary given one
sentence and ngram, we use prefix trie to restore the distribution for quick query:
    assume:
        vocab = 'x y'
        sentence = 'a b c'
        ngram = 3

    trie[0]:
        {'NULL': logP(x|NULL), logP(y|NULL)}
    trie[1]:
        {'a': [logP(x|a), logP(y|a)],
         'b': [logP(x|b), logP(y|b)],
         'c': [logP(x|c), logP(y|c)]}
    trie[2]:
        {'ab': [logP(x|ab), logP(y|ab)],
         'bc': [logP(x|bc), logP(y|bc)]}

    logP lists are sorted in descending order of log probabilities

##################################################################
'''


def get_ngram_vocab_prob(m, vocab, sent, ngram, ltrie):
    # ngram > 1
    lsent = sent if type(sent) is list else sent.split()

    ldic = []
    # 0, 1, 2, ..., ngram - 1
    for i in xrange(ngram):
        ldic.append({})

    state_in = kenlm.State()
    m.NullContextWrite(state_in)
    # Use <s> as context.  If you don't want <s>, use m.NullContextWrite(state).
    # m.BeginSentenceWrite(ngram_state)
    probs = []
    dist = {}
    for v in vocab:
        state_out = kenlm.State()
        full_score = m.BaseFullScore(state_in, v, state_out)
        # print full_score.log_prob, full_score.ngram_length, full_score.oov
        #probs.append((full_score.log_prob, full_score.ngram_length, full_score.oov, v))
        dist[v] = (full_score.log_prob, full_score.ngram_length, full_score.oov)
    # given 0 word, probs
    # probs.sort(reverse=True)    # lg->sm
    ldic[0]['null'] = trie(dist)

    for wid in range(len(lsent)):
        prev_words = lsent[wid - (ngram - 2) if wid - (ngram - 2) >= 0 else 0:wid + 1]
        gram_m1 = len(prev_words)
        for i in range(1, gram_m1 + 1):
            l_gram_prev = prev_words[-i:]
            s_gram_prev = ' '.join(l_gram_prev)
            # print s_gram_prev
            if not s_gram_prev in ldic[i]:
                probs = []
                state_in = kenlm.State()
                m.NullContextWrite(state_in)
                for w in l_gram_prev:
                    # print w, l_gram_prev
                    ngram_state = kenlm.State()
                    full_score = m.BaseFullScore(state_in, w, ngram_state)
                    # print w
                    # print full_score
                    state_in = ngram_state

                s = time.time()
                for v in vocab:
                    state_out = kenlm.State()
                    full_score = m.BaseFullScore(ngram_state, v, state_out)
                    # print v
                    # print full_score
                    # full_score.ngram_length is the matched ngram length ending with v in
                    # (l_gram_prev + v)
                    #probs.append((full_score.log_prob, full_score.ngram_length, full_score.oov, v))
                    dist[v] = (full_score.log_prob, full_score.ngram_length, full_score.oov)

                print time.time() - s
                print 'add....', len(dist)
                # probs.sort(reverse=True)
                j = 0
                sq = time.time()
                print dist['wonderful']
                print time.time() - sq
                for k, v in dist.iteritems():
                    if j < 10:
                        print k, v
                    j += 1
                ldic[i][s_gram_prev] = trie(dist)

                sq = time.time()
                tdist = trie(dist)
                print 'create trie: ', time.time() - sq

                print tdist.longest_prefix('wandskafjkasdjfas')

                j = 0
                sq = time.time()
                print tdist['wonderful']
                print time.time() - sq
                for k, v in tdist.iteritems():
                    if j < 10:
                        print k, v
                    j += 1

    for i in xrange(ngram):
        ltrie.append(trie(ldic[i]))


def vocab_prob_given_ngram(m, v_prev_ngram_wid, trg_vocab, trg_vocab_i2w):
    probs, words = [], []

    vocab = trg_vocab.keys()
    if len(v_prev_ngram_wid) == 0 or v_prev_ngram_wid[0] == -1:
        print 'zw', v_prev_ngram_wid[0]
        state_in = kenlm.State()
        m.NullContextWrite(state_in)
        for v in vocab:
            state_out = kenlm.State()
            log_prob = m.BaseScore(state_in, v, state_out)
            probs.append(log_prob)
            words.append(trg_vocab[v])
        return probs, words

    v_prev_ngram_w = []
    for wid in v_prev_ngram_wid:
        print '---------', trg_vocab_i2w[wid]
        v_prev_ngram_w.append(trg_vocab_i2w[wid])

    state_in = kenlm.State()
    m.NullContextWrite(state_in)
    for w in v_prev_ngram_w:
        ngram_state = kenlm.State()
        m.BaseScore(state_in, w, ngram_state)
        state_in = ngram_state

    for v in vocab:
        state_out = kenlm.State()
        log_prob = m.BaseScore(ngram_state, v, state_out)
        probs.append(log_prob)
        words.append(v)

    return probs, words


if __name__ == '__main__':

    vocab = ['i', 'am', 'a', 'good', 'student', '.', 'every', 'has', 'have']

    #test = 'every cloud has a silver lining .'
    test = 'it is good weather today , doesn\'t it ?'
    m = kenlm.Model('3gram.lc-tok.klm')
    #m = kenlm.Model('input.txt.arpa')

    sys.stderr.write('use {}-gram langauge model\n'.format(m.order))

    ngram = 4
    ngram = m.order if ngram > m.order else ngram

    ltrie = []
    get_ngram_vocab_prob(m, vocab, test, ngram, ltrie)

    for t in ltrie:
        for i, (k, v) in enumerate(t.iteritems()):
            print k, i
            for vv in v:
                print vv
