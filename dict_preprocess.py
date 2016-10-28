#!/usr/bin/env python

import argparse
import cPickle
import gzip
import bz2
import logging
import os

import numpy
import tables

from collections import Counter
from operator import add
from numpy.lib.stride_tricks import as_strided

parser = argparse.ArgumentParser(
    description="""
This takes a list of .txt or .txt.gz files and does word counting and
creating a dictionary (potentially limited by size). It uses this
dictionary to binarize the text into a numeric format (replacing OOV
words with 1) and create n-grams of a fixed size (padding the sentence
with 0 for EOS and BOS markers as necessary). The n-gram data can be
split up in a training and validation set.

The n-grams are saved to HDF5 format whereas the dictionary, word counts
and binarized text are all pickled Python objects.
""", formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("input", type=argparse.FileType('r'), nargs="+",
                    help="The input files")
parser.add_argument("-b", "--binarized-text", default='binarized_text.pkl',
                    help="the name of the pickled binarized text file")
parser.add_argument("-d", "--dictionary", default='vocab.pkl',
                    help="the name of the pickled binarized text file")
parser.add_argument("-n", "--ngram", type=int, metavar="N",
                    help="create n-grams")
parser.add_argument("-v", "--vocab", type=int, metavar="N",
                    help="limit vocabulary size to this number, which must "
                          "include BOS/EOS and OOV markers")
parser.add_argument("-p", "--pickle", action="store_true",
                    help="pickle the text as a list of lists of ints")
parser.add_argument("-s", "--split", type=float, metavar="N",
                    help="create a validation set. If >= 1 take this many "
                         "samples for the validation set, if < 1, take this "
                         "fraction of the samples")
parser.add_argument("-o", "--overwrite", action="store_true",
                    help="overwrite earlier created files, also forces the "
                         "program not to reuse count files")
parser.add_argument("-e", "--each", action="store_true",
                    help="output files for each separate input file")
parser.add_argument("-c", "--count", action="store_true",
                    help="save the word counts")
parser.add_argument("-t", "--char", action="store_true",
                    help="character-level processing")
parser.add_argument("-l", "--lowercase", action="store_true",
                    help="lowercase")

parser.add_argument("-top", "--topkwords", type=int,
                    help="generate pickled top k dictionary")
parser.add_argument("-rl", "--realvocab", default="vocab.rl",
                    help="the real mapping between words and id")


def open_files():
    base_filenames = []
    for i, input_file in enumerate(args.input):
        print i, input_file, input_file.name
        dirname, filename = os.path.split(input_file.name)
        if filename.split(os.extsep)[-1] == 'gz':
            base_filename = filename.rstrip('.gz')  # remove the .gz suffix
        elif filename.split(os.extsep)[-1] == 'bz2':
            base_filename = filename.rstrip('.bz2')  # remove the .bz2 suffix
        else:
            base_filename = filename
        if base_filename.split(os.extsep)[-1] == 'txt':
            # after remove the .gz or .bz2 suffix, we continue to remove .txt suffix
            base_filename = base_filename.rstrip('.txt')
        if filename.split(os.extsep)[-1] == 'gz':
            args.input[i] = gzip.GzipFile(input_file.name, input_file.mode,
                                          9, input_file)
        elif filename.split(os.extsep)[-1] == 'bz2':
            args.input[i] = bz2.BZ2File(input_file.name, input_file.mode)
        base_filenames.append(base_filename)
    return base_filenames   # the base filename of the input text, which is 'train.src' here


def safe_pickle(obj, filename):
    if os.path.isfile(filename) and not args.overwrite:
        logger.warning("Not saving %s, already exists." % (filename))
    else:
        if os.path.isfile(filename):
            logger.info("Overwriting %s." % filename)
        else:
            logger.info("Saving to %s." % filename)
        with open(filename, 'wb') as f:
            # print obj
            cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)


def safe_hdf(array, name):
    if os.path.isfile(name + '.hdf') and not args.overwrite:
        logger.warning("Not saving %s, already exists." % (name + '.hdf'))
    else:
        if os.path.isfile(name + '.hdf'):
            logger.info("Overwriting %s." % (name + '.hdf'))
        else:
            logger.info("Saving to %s." % (name + '.hdf'))
        with tables.openFile(name + '.hdf', 'w') as f:
            atom = tables.Atom.from_dtype(array.dtype)
            filters = tables.Filters(complib='blosc', complevel=5)
            ds = f.createCArray(f.root, name.replace('.', ''), atom,
                                array.shape, filters=filters)
            ds[:] = array


def create_dictionary():
    # Part I: Counting the words
    counters = []
    sentence_counts = []
    global_counter = Counter()

    # actually we just extract dictionary from one file, there is only one
    # element in list 'base_filenames'
    for input_file, base_filename in zip(args.input, base_filenames):
        count_filename = base_filename + '.count.pkl'   # 'train.src.count.pkl'
        # print args.input    # list
        # print input_file    # element in args.input
        # print input_file.name
        # print type(input_file)
        # 'train.src', remove the path before file name
        input_filename = os.path.basename(input_file.name)
        if os.path.isfile(count_filename) and not args.overwrite:
            logger.info("Loading word counts for %s from %s"
                        % (input_filename, count_filename))
            with open(count_filename, 'rb') as f:
                counter = cPickle.load(f)
            sentence_count = sum([1 for line in input_file])
        else:   # the count file 'train.src.count.pkl' does not exist
            logger.info("Counting words in %s" % input_filename)
            counter = Counter()
            sentence_count = 0
            for line in input_file:
                if args.lowercase:
                    line = line.lower()
                words = None
                if args.char:   # read each line in source file char by char
                    words = list(line.strip().decode('utf-8'))
                else:
                    words = line.strip().split(' ')
                counter.update(words)   # Counter({'book':11, 'strange':2, ..., })
                global_counter.update(words)
                sentence_count += 1
        counters.append(counter)
        sentence_counts.append(sentence_count)
        logger.info("%d unique words in %d sentences with a total of %d words."
                    % (len(counter), sentence_count, sum(counter.values())))
        if args.each and args.count:
            safe_pickle(counter, count_filename)
        input_file.seek(0)

    # Part II: Combining the counts
    combined_counter = global_counter   # all Counters from all input files (we have one here)
    logger.info("Total: %d unique words in %d sentences with a total "
                "of %d words."
                % (len(combined_counter), sum(sentence_counts),
                   sum(combined_counter.values())))
    if args.count:
        safe_pickle(combined_counter, 'combined.count.pkl')

    # Part III: Creating the dictionary
    if args.vocab is not None:
        vocab_count = []
        if args.vocab <= 2:  # words number of vocabulary you want to generate is less than 2
            # we build vocabulary with all words
            logger.info(
                'You input vocab number less than 2 ? Building a dictionary with all unique words')
            # we add 3 here, because there are there symbols <s>, <UNK> and </s>
            args.vocab = len(combined_counter) + 3
            logger.info("Want encodes all words (%s) and %s (<s> <UNK> </s>)" %
                        (len(combined_counter), 3))
        else:
            logger.info("Want encodes %s most common words and %s (<s> <UNK> </s>)" %
                        ((args.vocab - 3), 3))
        # else:
        #    logger.info('Building a dictionary with %s unique words.' % args.vocab)
        #    vocab_count = combined_counter.most_common(args.vocab)

        # because there are <s>, <UNK> and </s>, they use two indexes 0 and 1, so we generate (args.vocab - 2) words first,
        # add <s>, <UNK> and </s>, it is just args.vocab words
        vocab_count = combined_counter.most_common(args.vocab - 3)  # this line is very important

        # vocab_count = combined_counter.most_common(args.vocab - 2)	#there is a problem here, less two words
        # print len(vocab_count)
        # print vocab_count
        # print sum([count for word, count in vocab_count])
        # print sum(combined_counter.values())
    else:
        logger.info("Creating dictionary of all words")
        vocab_count = counter.most_common()
        args.vocab = len(combined_counter) + 3
        logger.info('No number input.')
        logger.info('Want encodes all words (%s) and %s (<s> <UNK> </s>)' %
                    (len(combined_counter), 2))

    # print 'total words number in Counter: ' + str(sum([count for word, count in vocab_count]))
    # print 'total words number in input file: ' + str(sum(combined_counter.values()))
    actual_word_num = min((args.vocab - 3), len(combined_counter))
    logger.info("Actually, we create dictionary of %s indexes for most common words and "
                "3 indexes for (<s> <UNK> </s>)." % (actual_word_num))
    logger.info("Totally %s words, covering %2.8f%% of the text." % (actual_word_num + 3,
                                                                     100.0 * sum([count for word, count in vocab_count]) /
                                                                     sum(combined_counter.values())))  # coverage rate = (total words number in Counter) / (total words number in input file)
    vocab = {'UNK': 1, '<s>': 0, '</s>': 0}  # we will replace index of </s> by the (vocab_size - 1)
    # vocab = {'<s>': 0, '<UNK>': 1, '</s>': (args.vocab-1)}    # we will replace index of </s> by the (vocab_size - 1)
    # because we add start symbol'(0), '<UNK>'(1) and 'end symbol'(vocab_size
    # - 1), so word in vocabulary starts from index 2
    for i, (word, count) in enumerate(vocab_count):
        vocab[word] = i + 2     # note that the values of words (keys) are index, not count
    # at this moment, the index of <s>, <UNK> and </s> are 0, 1 and 2
    # there should be 30000 words in vocab (contains <s>, <UNK> and </s>)
    if args.realvocab:
        f = open(args.realvocab, 'w')
        for k, v in vocab.iteritems():
            f.write(k + ', ' + str(v) + '\n')
    safe_pickle(vocab, args.dictionary)

    if args.topkwords is not None:
        vocab_topk = {}
        vocab_topk_count = combined_counter.most_common(args.topkwords)  # top 50
        logger.info("Creating dictionary of %s most common words, covering "
                    "%2.1f%% of the text."
                    % (args.topkwords,
                       100.0 * sum([count for word, count in vocab_topk_count]) /
                       sum(combined_counter.values())))
        for i, (word, count) in enumerate(vocab_topk_count):  # the word is possibly '' !!!
            vocab_topk[word] = i + 2  # we need add 2 here to corresponding to the whole vocab !!!
        dirname = os.path.dirname(args.dictionary)  # ./data/
        pklname = os.path.basename(args.dictionary)  # vocab.zh-en.en.pkl
        vocab_topk_pkl = os.path.join(dirname, str(args.topkwords) + pklname)
        vocab_topk_dict_name = str(args.topkwords) + '.'.join(pklname.split('.')[:-1]) + '.dict'
        vocab_topk_dict = os.path.join(dirname, vocab_topk_dict_name)

        ftop = open(vocab_topk_dict, 'w')
        for k, v in vocab_topk.iteritems():
            ftop.write(k + ', ' + str(v) + '\n')

        safe_pickle(vocab_topk, vocab_topk_pkl)

    return combined_counter, sentence_counts, counters, vocab


def binarize():
    if args.ngram:
        assert numpy.iinfo(numpy.uint16).max > len(vocab)
        ngrams = numpy.empty((sum(combined_counter.values()) +
                              sum(sentence_counts), args.ngram),
                             dtype='uint16')
    binarized_corpora = []
    total_ngram_count = 0
    for input_file, base_filename, sentence_count in \
            zip(args.input, base_filenames, sentence_counts):
        input_filename = os.path.basename(input_file.name)
        logger.info("Binarizing %s." % (input_filename))
        binarized_corpus = []
        ngram_count = 0
        for sentence_count, sentence in enumerate(input_file):
            if args.lowercase:
                sentence = sentence.lower()
            if args.char:
                words = list(sentence.strip().decode('utf-8'))
            else:
                words = sentence.strip().split(' ')
            binarized_sentence = [vocab.get(word, 1) for word in words]
            binarized_corpus.append(binarized_sentence)
            if args.ngram:
                padded_sentence = numpy.asarray(
                    [0] * (args.ngram - 1) + binarized_sentence + [0]
                )
                ngrams[total_ngram_count + ngram_count:
                       total_ngram_count + ngram_count + len(words) + 1] =  \
                    as_strided(
                        padded_sentence,
                        shape=(len(words) + 1, args.ngram),
                        strides=(padded_sentence.itemsize,
                                 padded_sentence.itemsize)
                )
            ngram_count += len(words) + 1
        # endfor sentence in input_file
        # Output
        if args.each:
            if args.pickle:
                safe_pickle(binarized_corpus, base_filename + '.pkl')
            if args.ngram and args.split:
                if args.split >= 1:
                    rows = int(args.split)
                else:
                    rows = int(ngram_count * args.split)
                logger.info("Saving training set (%d samples) and validation "
                            "set (%d samples)."
                            % (ngram_count - rows, rows))
                rows = numpy.random.choice(ngram_count, rows, replace=False)
                safe_hdf(ngrams[total_ngram_count + rows],
                         base_filename + '_valid')
                safe_hdf(
                    ngrams[total_ngram_count + numpy.setdiff1d(
                        numpy.arange(ngram_count),
                        rows, True
                    )], base_filename + '_train'
                )
            elif args.ngram:
                logger.info("Saving n-grams to %s." % (base_filename + '.hdf'))
                safe_hdf(ngrams, base_filename)
        binarized_corpora += binarized_corpus
        total_ngram_count += ngram_count
        input_file.seek(0)
    # endfor input_file in args.input
    if args.pickle:
        safe_pickle(binarized_corpora, args.binarized_text)
    if args.ngram and args.split:
        if args.split >= 1:
            rows = int(args.split)
        else:
            rows = int(total_ngram_count * args.split)
        logger.info("Saving training set (%d samples) and validation set (%d "
                    "samples)."
                    % (total_ngram_count - rows, rows))
        rows = numpy.random.choice(total_ngram_count, rows, replace=False)
        safe_hdf(ngrams[rows], 'combined_valid')
        safe_hdf(ngrams[numpy.setdiff1d(numpy.arange(total_ngram_count),
                                        rows, True)], 'combined_train')
    elif args.ngram:
        safe_hdf(ngrams, 'combined')


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('preprocess')
    args = parser.parse_args()
    base_filenames = open_files()   # return ['train.src'] because no .txt suffix
    # print base_filenames
    combined_counter, sentence_counts, counters, vocab = create_dictionary()
    if args.ngram or args.pickle:
        binarize()
