#!/usr/bin/python

import argparse
import logging
import os
import subprocess
import tarfile
import urllib2
import uuid
import configurations

from picklable_itertools.extras import equizip

TRAIN_DATA_URL = 'http://www.statmt.org/wmt15/training-parallel-nc-v10.tgz'
VALID_DATA_URL = 'http://www.statmt.org/wmt15/dev-v2.tgz'
PREPROCESS_URL = 'https://raw.githubusercontent.com/lisa-groundhog/' +\
                 'GroundHog/master/experiments/nmt/preprocess/preprocess.py'
TOKENIZER_URL = 'https://raw.githubusercontent.com/moses-smt/mosesdecoder/' +\
                'master/scripts/tokenizer/tokenizer.perl'

TOKENIZER_PREFIXES = 'https://raw.githubusercontent.com/moses-smt/' +\
                     'mosesdecoder/master/scripts/share/nonbreaking_' +\
                     'prefixes/nonbreaking_prefix.'
BLEU_SCRIPT_URL = 'https://raw.githubusercontent.com/moses-smt/mosesdecoder' +\
                  '/master/scripts/generic/multi-bleu.perl'
OUTPUT_DIR = './data'
PREFIX_DIR = './share/nonbreaking_prefixes'

parser = argparse.ArgumentParser(
    description="""
This script donwloads parallel corpora given source and target pair language
indicators and preprocess it respectively for neural machine translation.For
the preprocessing, moses tokenizer is applied first then tokenized corpora
are used to extract vocabularies for source and target languages. Finally the
tokenized parallel corpora are shuffled for SGD.

Note that, this script is written specificaly for WMT15 training and
development corpora, hence change the corresponding sections if you plan to use
some other data.
""", formatter_class=argparse.RawTextHelpFormatter)
# parser.add_argument("-s", "--source", type=str, help="Source language",
#                    default="train.zh")
# parser.add_argument("-t", "--target", type=str, help="Target language",
#                    default="train.en")
parser.add_argument("--source-dev", type=str, default="newstest2013.cs",
                    help="Source language dev filename")
parser.add_argument("--target-dev", type=str, default="newstest2013.en",
                    help="Target language dev filename")
parser.add_argument("--source-vocab", type=int, default=30000,
                    help="Source language vocabulary size")
parser.add_argument("--target-vocab", type=int, default=30000,
                    help="Target language vocabulary size")


def download_and_write_file(url, file_name):
    logger.info("Downloading [{}]".format(url))
    if not os.path.exists(file_name):
        path = os.path.dirname(file_name)
        if not os.path.exists(path):
            os.makedirs(path)
        u = urllib2.urlopen(url)
        f = open(file_name, 'wb')
        meta = u.info()
        file_size = int(meta.getheaders("Content-Length")[0])
        logger.info("...saving to: %s Bytes: %s" % (file_name, file_size))
        file_size_dl = 0
        block_sz = 8192
        while True:
            buffer = u.read(block_sz)
            if not buffer:
                break
            file_size_dl += len(buffer)
            f.write(buffer)
            status = r"%10d  [%3.2f%%]" % \
                (file_size_dl, file_size_dl * 100. / file_size)
            status = status + chr(8) * (len(status) + 1)
            print status,
        f.close()
    else:
        logger.info("...file exists [{}]".format(file_name))


def extract_tar_file_to(file_to_extract, extract_into, names_to_look):
    extracted_filenames = []
    try:
        logger.info("Extracting file [{}] into [{}]"
                    .format(file_to_extract, extract_into))
        tar = tarfile.open(file_to_extract, 'r')
        src_trg_files = [ff for ff in tar.getnames()
                         if any([ff.find(nn) > -1 for nn in names_to_look])]
        if not len(src_trg_files):
            raise ValueError("[{}] pair does not exist in the archive!"
                             .format(src_trg_files))
        for item in tar:
            # extract only source-target pair
            if item.name in src_trg_files:
                file_path = os.path.join(extract_into, item.path)
                if not os.path.exists(file_path):
                    logger.info("...extracting [{}] into [{}]"
                                .format(item.name, file_path))
                    tar.extract(item, extract_into)
                else:
                    logger.info("...file exists [{}]".format(file_path))
                extracted_filenames.append(
                    os.path.join(extract_into, item.path))
    except Exception as e:
        logger.error("{}".format(str(e)))
    return extracted_filenames


def tokenize_text_files(files_to_tokenize, tokenizer):
    for name in files_to_tokenize:
        logger.info("Tokenizing file [{}]".format(name))
        out_file = os.path.join(
            OUTPUT_DIR, os.path.basename(name) + '.tok')
        logger.info("...writing tokenized file [{}]".format(out_file))
        var = ["perl", tokenizer,  "-l", name.split('.')[-1]]
        if not os.path.exists(out_file):
            with open(name, 'r') as inp:
                with open(out_file, 'w', 0) as out:
                    subprocess.check_call(
                        var, stdin=inp, stdout=out, shell=False)
        else:
            logger.info("...file exists [{}]".format(out_file))


def create_vocabularies(src_data, trg_data, preprocess_file, topk_trg_vocab, **kwargs):
    tr_files = [src_data, trg_data]
    source_suffix = tr_files[0].split('.')[-1]
    target_suffix = tr_files[1].split('.')[-1]
    src_vocab_name = os.path.join(
        OUTPUT_DIR, 'vocab.{}-{}.{}.pkl'.format(
            source_suffix, target_suffix, source_suffix))
    trg_vocab_name = os.path.join(
        OUTPUT_DIR, 'vocab.{}-{}.{}.pkl'.format(
            source_suffix, target_suffix, target_suffix))
    src_filename = os.path.basename(
        tr_files[[i for i, n in enumerate(tr_files)
                  if n.endswith(source_suffix)][0]])    # remove previous path: /zhang/wen -> wen
    trg_filename = os.path.basename(
        tr_files[[i for i, n in enumerate(tr_files)
                  if n.endswith(target_suffix)][0]])
    # dict
    src_dict_name = os.path.join(
        OUTPUT_DIR, 'vocab.{}-{}.{}.dict'.format(
            source_suffix, target_suffix, source_suffix))
    trg_dict_name = os.path.join(
        OUTPUT_DIR, 'vocab.{}-{}.{}.dict'.format(
            source_suffix, target_suffix, target_suffix))

    logger.info("Creating source vocabulary [{}]".format(src_vocab_name))
    if not os.path.exists(src_vocab_name):
        subprocess.check_call(" python {} -d {} -v {} -rl {} {}".format(
            preprocess_file, src_vocab_name, args.source_vocab,
            src_dict_name, os.path.join(OUTPUT_DIR, src_filename)),
            shell=True)
    else:
        logger.info("...file exists [{}]".format(src_vocab_name))
        '''
        subprocess.check_call(" python {} -o -d {} -v {} -rl {} {}".format(
            preprocess_file, src_vocab_name, args.source_vocab,
            src_dict_name, os.path.join(OUTPUT_DIR, src_filename)),
            shell=True)
        '''

    logger.info("Creating target vocabulary [{}]".format(trg_vocab_name))
    if not os.path.exists(trg_vocab_name):
        subprocess.check_call(" python {} -d {} -v {} -rl {} -top {} {}".format(
            preprocess_file, trg_vocab_name, args.target_vocab,
            trg_dict_name, topk_trg_vocab, os.path.join(OUTPUT_DIR, trg_filename)),
            shell=True)
    else:
        logger.info("...file exists [{}]".format(trg_vocab_name))
        '''
        subprocess.check_call(" python {} -o -d {} -v {} -rl {} -top {} {}".format(
            preprocess_file, trg_vocab_name, args.target_vocab,
            trg_dict_name, topk_trg_vocab, os.path.join(OUTPUT_DIR, trg_filename)),
            shell=True)
        '''
    # return src_filename, trg_filename, aln_filename


def merge_parallel(src_filename, trg_filename, aln_filename, merged_filename):
    try:
        with open(aln_filename, 'r') as align, open(src_filename, 'r') as left, open(trg_filename, 'r') as right, open(merged_filename, 'w') as final:
            for lline, rline, aline in equizip(left, right, align):
                if (lline != '\n') and (rline != '\n') and (aline != '\n'):
                    final.write(lline[:-1] + ' ||| ' + rline[:-1] + ' ||| ' + aline)
    except IOError:
        print 'no align file'
        with open(src_filename, 'r') as left, open(trg_filename, 'r') as right, open(merged_filename, 'w') as final:
            for lline, rline in equizip(left, right):
                if (lline != '\n') and (rline != '\n'):
                    final.write(lline[:-1] + ' ||| ' + rline)


def split_parallel(merged_filename, src_filename, trg_filename, aln_filename):
    try:
        with open(merged_filename, 'r') as combined, open(src_filename, 'w') as left, open(trg_filename, 'w') as right, open(aln_filename, 'w') as align:
            for line in combined:
                line = line.split('|||')
                left.write(line[0].strip() + '\n')
                right.write(line[1].strip() + '\n')
                align.write(line[2].strip() + '\n')
    except IOError:
        print 'no align file'
        with open(merged_filename) as combined, open(src_filename, 'w') as left, open(trg_filename, 'w') as right:
            for line in combined:
                line = line.split('|||')
                left.write(line[0].strip() + '\n')
                right.write(line[1].strip() + '\n')


def shuffle_parallel(src_filename, trg_filename):
    logger.info("Shuffling jointly [{}] & [{}]".format(src_filename, trg_filename))
    out_src = src_filename + '.shuf'
    out_trg = trg_filename + '.shuf'
    merged_filename = str(uuid.uuid4())
    shuffled_filename = str(uuid.uuid4())
    if not os.path.exists(out_src) or not os.path.exists(out_trg):
        # overwrite all files even one does not exsit
        try:
            merge_parallel(src_filename, trg_filename, '',  merged_filename)
            subprocess.check_call(
                " shuf {} > {} ".format(merged_filename, shuffled_filename),
                shell=True)
            split_parallel(shuffled_filename, out_src, out_trg, '')
            logger.info(
                "...files shuffled [{}] & [{}]".format(out_src, out_trg))
        except Exception as e:
            logger.error("{}".format(str(e)))
    else:
        logger.info("...files exist [{}] & [{}]".format(out_src, out_trg))
    if os.path.exists(merged_filename):
        os.remove(merged_filename)
    if os.path.exists(shuffled_filename):
        os.remove(shuffled_filename)


def main():
    config = getattr(configurations, 'get_config_cs2en')()

    OUTPUT_DIR = config['datadir']
    #preprocess_file = os.path.join(OUTPUT_DIR, 'preprocess.py')
    #bleuscore_file = os.path.join(OUTPUT_DIR, 'multi-bleu.perl')

    # Download the News Commentary v10 ~122Mb and extract it

    # Download bleu score calculation script
    #download_and_write_file(BLEU_SCRIPT_URL, bleuscore_file)

    # Download preprocessing script
    #download_and_write_file(PREPROCESS_URL, preprocess_file)

    # Download tokenizer

    # Apply preprocessing and construct vocabularies
    #tr_files = ["trn.src", "trn.trg"]
    create_vocabularies(**config)

    # Shuffle datasets
    shuffle_parallel(config['src_data'], config['trg_data'])

    from manvocab import write_tr_target_word_into_file, valid_sent_target_vcab_set

    write_tr_target_word_into_file(**config)
    valid_sent_target_vcab_set(**config)


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('prepare_data')

    args = parser.parse_args()
    main()
