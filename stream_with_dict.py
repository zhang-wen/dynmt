# -*- coding: utf-8 -*-
import numpy

from fuel.datasets import TextFile
from fuel.schemes import ConstantScheme
from fuel.streams import DataStream
from fuel.transformers import (
    Merge, Batch, Filter, Padding, SortMapping, Unpack, Mapping)

from six.moves import cPickle

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def ensure_special_tokens(vocab, bos_idx=0, eos_idx=0, unk_idx=1):
    """Ensures special tokens exist in the dictionary."""

    # remove tokens if they exist in some other index
    # in vocab (dict), the v (index) of bos, eos, unk are 0, -1, and 1
    tokens_to_remove = [k for k, v in vocab.items()
                        if v in [bos_idx, eos_idx, unk_idx, -1]]    # eos_idx is -1 in vocab, we remove it
    for token in tokens_to_remove:
        vocab.pop(token)    # remove the bos, unk and eos in vocabulary
    # put corresponding item
    vocab['<S>'] = bos_idx      # 0
    vocab['</S>'] = eos_idx     # 29999
    vocab['<UNK>'] = unk_idx    # 1
    # for k, v in vocab.iteritems():
    #    print k, v
    return vocab    # final vocabulary


def _length(sentence_pair):
    """Assumes target is the second element in the tuple."""
    # sort by target sentence length
    return len(sentence_pair[1])


class PaddingWithEOS(Padding):
    """Padds a stream with given end of sequence idx."""

    def __init__(self, data_stream, bos_idx, eos_idx, **kwargs):
        # print eos_idx   # [29999, 29999]
        kwargs['data_stream'] = data_stream
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        super(PaddingWithEOS, self).__init__(**kwargs)

    def transform_batch(self, batch):
         # if you use next() function here, the final data_stream contain data every other batch, i do not know why doing like this
        # batch = next(self.child_epoch_iterator)	# i annote here !!!!!!!!!!!!!!!!!
        data = list(batch)
        data_with_masks = []
        self.mask_sources = ('source', 'target')
        # print self.mask_sources     # ('source', 'target', 'dict')
        for i, (source, source_data) in enumerate(
                zip(self.data_stream.sources, data)):
            # 0 source [[0, 1, 29999], [0,2,3,29999]]
            # 1 target [[0, 1, 29999], [0,2,3,29999]]
            # 2 dict [[0, 1, 29999], [0,2,3,29999]]
            # print i, source, source_data
            if source not in self.mask_sources:
                data_with_masks.append(source_data)     # do not add mask for this source
                continue

            # sample: [    0,  1108,    66,     1, 29999]
            # numpy.asarray(sample) is a numpy.ndarray, shape is (5,), shape[0] == 5
            # batch_size = 5, [(5,), (20,), (33,), (13,), (9,0)]
            shapes = [numpy.asarray(sample).shape for sample in source_data]
            lengths = [shape[0] for shape in shapes]    # [5, 20, 33, 13, 9]
            max_sequence_length = max(lengths)          # 33
            rest_shape = shapes[0][1:]          # here is ()
            if not all([shape[1:] == rest_shape for shape in shapes]):
                raise ValueError("All dimensions except length must be equal")
            dtype = numpy.asarray(source_data[0]).dtype

            '''
            padded_data = numpy.ones((len(source_data), max_sequence_length) + rest_shape,
                                     dtype=dtype) * self.eos_idx[i]
            # all array([[29999, 29999, 29999, 29999, 29999, 29999, 29999, 29999, 29999, ... , ]
            '''

            padded_data = numpy.ones(
                (len(source_data), max_sequence_length) + rest_shape,
                dtype=dtype) * self.bos_idx[i]
            # all array([[0, 0, 0, 0, 0, 0, 0, 0, 0, ... , ]

            for i, sample in enumerate(source_data):
                # assign the real indexes value of the sentence, at the first len(sample) locations
                padded_data[i, :len(sample)] = sample
            data_with_masks.append(padded_data)

            # mask = numpy.zeros((len(source_data), max_sequence_length),
            #                   self.mask_dtype)  # all array([[0, 0, 0, 0, 0, 0, 0, 0, 0, ... , ]
            mask = numpy.zeros((len(source_data), max_sequence_length),
                               dtype='int32')  # all array([[0, 0, 0, 0, 0, 0, 0, 0, 0, ... , ]
            for i, sequence_length in enumerate(lengths):
                # assign value 1 at the first len(sample) locations
                mask[i, :sequence_length] = 1
            data_with_masks.append(mask)
        return tuple(data_with_masks)


class _oov_to_unk(object):
    """Maps out of vocabulary token index to unk token index."""

    def __init__(self, src_vocab_size=30000, trg_vocab_size=30000,
                 unk_id=1):
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size
        self.unk_id = unk_id

    def __call__(self, sentence_pair):
        '''
        return ([x if x < self.src_vocab_size else self.unk_id
                 for x in sentence_pair[0]],
                [x if x < self.trg_vocab_size else self.unk_id
                 for x in sentence_pair[1]])
        '''
        return ([x if x < self.src_vocab_size else self.unk_id
                 for x in sentence_pair[0]],
                [x if x < self.trg_vocab_size else self.unk_id
                 for x in sentence_pair[1]],
                [x if x < self.trg_vocab_size else self.unk_id
                 for x in sentence_pair[2]])    # add dict


class _too_long(object):
    """Filters sequences longer than given sequence length."""

    def __init__(self, seq_len=50):
        self.seq_len = seq_len

    def __call__(self, sentence_pair):
        # include the padding length 0 and 29999
        # print [len(sentence) <= self.seq_len  for sentence in sentence_pair] #[True, False]
        # return all([len(sentence) <= self.seq_len
        #            for sentence in sentence_pair])
        # both length of source sentence and target sentence are less than seq_len
        return all([len(sentence_pair[0]) <= self.seq_len, len(sentence_pair[1]) <= self.seq_len])


def get_tr_stream(src_vocab, trg_vocab, src_data, trg_data, dict_data,
                  src_vocab_size=30000, trg_vocab_size=30000, unk_id=1,
                  seq_len=50, batch_size=80, sort_k_batches=12, **kwargs):
    """Prepares the training data stream."""

    # Load dictionaries and ensure special tokens exist

    '''
    actual_src_vocab_num = len(src_vocab)
    actual_trg_vocab_num = len(trg_vocab)
    src_vocab = ensure_special_tokens(
        src_vocab if isinstance(src_vocab, dict)
        else cPickle.load(open(src_vocab)),
        bos_idx=0, eos_idx=(actual_src_vocab_num - 1) if
	actual_src_vocab_num - 3 <
	src_vocab_size else (src_vocab_size + 3 -
		1), unk_idx=unk_id)
    trg_vocab = ensure_special_tokens(
        trg_vocab if isinstance(trg_vocab, dict) else
        cPickle.load(open(trg_vocab)),
        bos_idx=0, eos_idx=(actual_trg_vocab_num - 1) if
	actual_trg_vocab_num - 3 < trg_vocab_size else
	(trg_vocab_size + 3 - 1), unk_idx=unk_id)
    '''

    src_vocab = ensure_special_tokens(
        src_vocab if isinstance(src_vocab, dict)
        else cPickle.load(open(src_vocab)),
        bos_idx=0, eos_idx=src_vocab_size - 1, unk_idx=unk_id)
    trg_vocab = ensure_special_tokens(
        trg_vocab if isinstance(trg_vocab, dict) else
        cPickle.load(open(trg_vocab)),
        bos_idx=0, eos_idx=trg_vocab_size - 1, unk_idx=unk_id)

    # for example:
    # source: 第五 章 罚则
    # target: chapter v penalty regulations
    # Get text files from both source and target
    src_dataset = TextFile([src_data], src_vocab)
    trg_dataset = TextFile([trg_data], trg_vocab)
    dict_dataset = TextFile([dict_data], trg_vocab)
    # for data in DataStream(src_dataset).get_epoch_iterator():
    #    print(data)     # looks like: ([0, 1649, 1764, 7458, 29999],)

    # Merge them to get a source, target pair
    stream = Merge([src_dataset.get_example_stream(),
                    trg_dataset.get_example_stream(),
                    dict_dataset.get_example_stream()],
                   ('source', 'target', 'dict'))    # data_stream.sources = 'source' or 'target'
    '''
    print 'init \n'
    num_before_filter = 0
    for data in stream.get_epoch_iterator():
        num_before_filter = num_before_filter + 1
        # print(data)
    '''
    # looks like: ([0, 1649, 1764, 7458, 29999], [0, 2662, 9329, 968, 200, 29999])

    # Filter sequences that are too long
    # Neither source sentence or target sentence can beyond the length seq_len
    # the lenght include the start symbol <s> and the end symbol </s>, so the actual sentence
    # length can not beyond (seq_len - 2)
    stream = Filter(stream,
                    predicate=_too_long(seq_len=seq_len))
    '''
    num_after_filter = 0
    # print 'after filter ... \n'
    for data in stream.get_epoch_iterator():
        num_after_filter = num_after_filter + 1
        # print(data)

    logger.info('\tby filtering, sentence-pairs from {} to {}.'.format(num_before_filter, num_after_filter))
    logger.info('\tfilter {} sentence-pairs whose source or target sentence exceeds {} words'.format(
        (num_before_filter - num_after_filter), seq_len))
    '''

    # Replace out of vocabulary tokens with unk token
    stream = Mapping(stream,
                     _oov_to_unk(src_vocab_size=src_vocab_size,
                                 trg_vocab_size=trg_vocab_size,
                                 unk_id=unk_id))    # do not need
    '''
    print 'after mapping unk ...'
    for data in stream.get_epoch_iterator():
        print(data)
    '''

    # still looks like: ([0, 1649, 1764, 7458, 29999], [0, 2662, 9329, 968, 200, 29999])
    # Build a batched version of stream to read k batches ahead
    # do not sort on the whole training data, first split the training data into several blocks,
    # each block contain (batch_size*sort_k_batches) sentence-pairs, we juse sort in each block,
    # finally, i understand !!!!!!!
    # remainder
    stream = Batch(stream,
                   iteration_scheme=ConstantScheme(
                       batch_size * sort_k_batches))

    '''
    print 'after sorted batch ... '
    for data in stream.get_epoch_iterator():
        print(data)
    '''

    # Sort all samples in the read-ahead batch
    # sort by the length of target sentence in (batch_size*sort_k_batches)
    # list for all training data, speed up
    stream = Mapping(stream, SortMapping(_length))
    '''
    print 'after sort ... '
    for data in stream.get_epoch_iterator():
        print(data)
    '''

    # Convert it into a stream again
    stream = Unpack(stream)
    '''
    print 'after unpack ... '
    for data in stream.get_epoch_iterator():
        print(data)
    '''
    # still looks like: ([0, 1649, 1764, 7458, 29999], [0, 2662, 9329, 968, 200, 29999])

    # remove the remainder ?
    # Construct batches from the stream with specified batch size
    stream = Batch(
        stream, iteration_scheme=ConstantScheme(batch_size))

    # after sort, each batch has batch_size sentence pairs
    '''
    print 'after final batch ... '
    i = 0
    for data in stream.get_epoch_iterator():
        i = i + 1
        print(data)
    print 'batchs: ', i
    '''

    # Pad sequences that are short
    masked_stream = PaddingWithEOS(
        stream, bos_idx=[0, 0, 0], eos_idx=[src_vocab_size - 1, trg_vocab_size - 1, trg_vocab_size - 1])
    # print 'after padding with mask ...'
    return masked_stream

'''
def get_dev_stream(val_set=None, src_vocab=None, src_vocab_size=30000,
                   unk_id=1, **kwargs):
    """Setup development set stream if necessary."""
    dev_stream = None
    if val_set is not None and src_vocab is not None:
        src_vocab = ensure_special_tokens(
            src_vocab if isinstance(src_vocab, dict) else
            cPickle.load(open(src_vocab)),
            bos_idx=0, eos_idx=src_vocab_size - 1, unk_idx=unk_id)
        dev_dataset = TextFile([val_set], src_vocab)
        dev_stream = DataStream(dev_dataset)    # no filter, no padding, no mask for dev_dataset
    return dev_stream
'''


# validation set, no batch, not padding
def get_dev_stream(val_set=None, valids_dict=None, src_vocab=None, trg_vocab=None,
                   src_vocab_size=30000, trg_vocab_size=30000, unk_id=1, **kwargs):
    """Setup development set stream if necessary."""

    dev_stream = None
    if val_set is not None and src_vocab is not None:
        # Load dictionaries and ensure special tokens exist
        src_vocab = ensure_special_tokens(
            src_vocab if isinstance(src_vocab, dict) else
            cPickle.load(open(src_vocab)),
            bos_idx=0, eos_idx=src_vocab_size - 1, unk_idx=unk_id)

        trg_vocab = ensure_special_tokens(
            trg_vocab if isinstance(trg_vocab, dict) else
            cPickle.load(open(trg_vocab)),
            bos_idx=0, eos_idx=trg_vocab_size - 1, unk_idx=unk_id)

        dev_dataset = TextFile([val_set], src_vocab, None)
        dev_dictset = TextFile([valids_dict], trg_vocab, None)
        #dev_stream = DataStream(dev_dataset)
        # Merge them to get a source, target pair
        dev_stream = Merge([dev_dataset.get_example_stream(),
                            dev_dictset.get_example_stream()],
                           ('source', 'valid_sent_trg_dict'))
    return dev_stream


if __name__ == '__main__':
    import configurations
    configuration = getattr(configurations, 'get_config_cs2en')()
    trstream = get_tr_stream(**configuration)
    for data in trstream.get_epoch_iterator():
        print data
        # print type(data[0]) # numpy.ndarray
        # print type(data)   # data is a tuple

        # take the batch sizes 3 as an example:
        # tuple: tuple[0] is indexes of source sentence (numpy.ndarray)
        # like array([[0, 23, 3, 4, 29999], [0, 2, 1, 29999], [0, 31, 333, 2, 1, 29999]])
        # tuple: tuple[1] is indexes of source sentence mask (numpy.ndarray)
        # like array([[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1]])
        # tuple: tuple[2] is indexes of target sentence (numpy.ndarray)
        # tuple: tuple[3] is indexes of target sentence mask (numpy.ndarray)
        # print numpy.asarray(data)
        # break
    dev_stream = get_dev_stream(**configuration)
    # for data in dev_stream.get_epoch_iterator():
    # print data
    # print numpy.asarray(data)
    # print data[0]   # such as [0, 294, 430, 45, 1009, 560, 708, 283, 437, 1, 164, 321, 29999]
    # print type(data[0]) # list
    # print type(data)    # tuple: ([0, 294, 430, 45, 1009, 560, 708, 283,
    # 437, 1, 164, 321, 29999],)
