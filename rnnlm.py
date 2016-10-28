import random
from collections import defaultdict
from itertools import count
import sys
#from G.et import *
#import G.et as dy
#import _gdynet as G
import _gdynet as G
import cPickle as pickle

LAYERS = 2
INPUT_DIM = 50
HIDDEN_DIM = 50

characters = list("abcdefghijklmnopqrstuvwxyz ")
#VOCAB_SIZE = len(characters)

vocabs = pickle.load(open('test-vocab/vocab.zh-en.en.pkl'))
characters.append("<EOS>")

char2int = {c: i for i, c in enumerate(vocabs)}
char2int['<EOS>'] = len(char2int) + 1
char2int['<UNK>'] = len(char2int) + 1
int2char = {index: word for word, index in char2int.iteritems()}
VOCAB_SIZE = len(char2int)
print 'vocab size:' , VOCAB_SIZE


model = G.Model()
srnn = G.SimpleRNNBuilder(LAYERS, INPUT_DIM, HIDDEN_DIM, model)
lstm = G.LSTMBuilder(LAYERS, INPUT_DIM, HIDDEN_DIM, model)

params = {}
params["lookup"] = model.add_lookup_parameters((VOCAB_SIZE, INPUT_DIM))
params["R"] = model.add_parameters((VOCAB_SIZE, HIDDEN_DIM))
params["bias"] = model.add_parameters((VOCAB_SIZE))

# return compute loss of RNN for one sentence


def do_one_sentence(rnn, sentence):
    # setup the sentence
    G.renew_cg()
    s0 = rnn.initial_state()
    R = G.parameter(params['R'])
    bias = G.parameter(params['bias'])    # _G.et.Expression
    lookup = params['lookup']   # _G.et.LookupParameters
    sentence = ['<EOS>'] + sentence.split() + ['<EOS>']
    sentence = [char2int[c] if c in char2int else (len(char2int)-1) for c in sentence]
    s = s0
    loss = []
    for char, next_char in zip(sentence, sentence[1:]):
        print char, next_char, lookup[char].npvalue().shape
        s = s.add_input(lookup[char])
        print s.s()[1].npvalue().shape
        probs = G.softmax(R * s.output() + bias)
        # Picking values from vector expressions
        # e = pick(e1, k)              # k is unsigned integer, e1 is vector. return e1[k]
        loss.append(-G.log(G.pick(probs, next_char)))   # probs[next_char]: ce
    loss = G.esum(loss)
    return loss

# generate from model:


def generate(rnn):
    def sample(probs):
        rnd = random.random()
        for i, p in enumerate(probs):
            rnd -= p
            if rnd <= 0:
                break
        return i

    # setup the sentence
    G.renew_cg()
    s0 = rnn.initial_state()
    R = G.parameter(params['R'])
    bias = G.parameter(params['bias'])
    lookup = params['lookup']

    s = s0.add_input(lookup[char2int['<EOS>']])
    out = []
    while True:
        probs = G.softmax(R * s.output() + bias)
        list_probs = probs.value()
        wid = sample(list_probs)
        w = int2char[wid]
        out.append(w)
        if out[-1] == '<EOS>':
            break
        next_char_emb = lookup[wid]
        s = s.add_input(next_char_emb)
    return ''.join(out[:-1])  # strip the <EOS>

def train(rnn, sentence):
    trainer = G.SimpleSGDTrainer(model)
    for i in xrange(200):
        loss = do_one_sentence(rnn, sentence)
        loss_value = loss.value()
        loss.backward()
        trainer.update()
        if i % 5 == 0:
            print loss_value,
            print generate(rnn)

if __name__ == '__main__':
    sentence = "a quick brown fox jumped over the lazy dog a quick brown fox jumped over the lazy dog"
    train(lstm, sentence)
