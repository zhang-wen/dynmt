from __future__ import print_function

import numpy
import copy
import os
import sys
import subprocess
import re
from utils import exeTime
import collections

from search_mle import mle_trans
from search_bs import beam_search_trans
from search_cp import cube_prune_trans
from search_naive import original_trans
from utils import _index2sentence, _filter_reidx


def fetch_bleu_from_file(fbleufrom):
    fread = open(fbleufrom, 'r')
    result = fread.readlines()
    fread.close()
    f_bleu = 0.
    f_multibleu = 0.
    for line in result:
        bleu_pattern = re.search(r'BLEU score = (0\.\d+)', line)
        if bleu_pattern:
            s_bleu = bleu_pattern.group(1)
            f_bleu = format(float(s_bleu) * 100, '0.2f')
        multi_bleu_pattern = re.search(r'BLEU = (\d+\.\d+)', line)
        if multi_bleu_pattern:
            s_multibleu = multi_bleu_pattern.group(1)
            f_multibleu = format(float(s_multibleu), '0.2f')
    return f_bleu, f_multibleu


def append_file(file_prefix, content):
    f = open(file_prefix, 'a')
    f.write(content)
    f.write('\n')
    f.close()


def valid_bleu(eval_dir, valid_out, eidx, uidx, beam, mode, kl):    # valid_out: valids/trans
    import configurations
    config = getattr(configurations, 'get_config_cs2en')()
    save_log = '{}.{}'.format(valid_out, 'log')
    child = subprocess.Popen('sh bleu.sh ../{} {} ../{} {}'.format(valid_out, config['val_prefix'],
                                                                   save_log, config['val_tst_dir']),
                             cwd=eval_dir, shell=True, stdout=subprocess.PIPE)
    bleu_out = child.communicate()
    child.wait()
    mteval_bleu, multi_bleu = fetch_bleu_from_file(save_log)
    # ori_mteval_bleu, ori_multi_bleu = fetch_bleu_from_file(oriref_bleu_log)
    sfig = '{}.{}'.format(config['val_set_out'], 'sfig')
    # sfig_content = str(eidx) + ' ' + str(uidx) + ' ' + str(mteval_bleu) + ' ' + \
    #    str(multi_bleu) + ' ' + str(ori_mteval_bleu) + ' ' + str(ori_multi_bleu)
    sfig_content = str(mode) + ' ' + str(beam) + ' ' + str(kl) + ' ' + \
        str(mteval_bleu) + ' ' + str(multi_bleu)
    append_file(sfig, sfig_content)

    os.rename(valid_out, "{}_{}_{}.txt".format(valid_out, mteval_bleu, multi_bleu))
    return mteval_bleu  # we use the bleu without unk and mteval-v11, process reference


def trans_onesent(s, n, switchs, mode, trg_vocab_i2w, k, kl, lm=None, tvcb=None, ptv=None):
    if mode == 0:
        trans = mle_trans(s, n, switchs, trg_vocab_i2w, k=k)
    elif mode == 1:
        trans = original_trans(s, n, switchs, trg_vocab_i2w, k=k)
    elif mode == 2:
        trans = beam_search_trans(s, n, switchs, trg_vocab_i2w, k=k, ptv=ptv)
    elif mode == 3:
        trans = cube_prune_trans(s, n, switchs, trg_vocab_i2w, k=k, kl=kl, lm=lm, tvcb=tvcb)
    return trans


def trans_samples(srcs, trgs, network, switchs, src_vocab_i2w, trg_vocab_i2w, bos_id, eos_id, k=10, mode=0, kl=0.,
                  lm=None, tvcb=None, ptv=None):
    for index in range(len(srcs)):
        # print 'before filter: '
        # print s[index]
        # [   37   785   600    44   160  4074   152  3737     2   399  1096   170      4     8    29999     0     0     0     0     0     0     0]
        s_filter = filter(lambda x: x != 0, srcs[index])
        print ('[{:3}] {}'.format('src', _index2sentence(s_filter, src_vocab_i2w)))
        # ndarray -> list
        # s_filter: [   37   785   600    44   160  4074   152  3737     2   399
        # 1096   170      4 8    29999]
        t_filter = filter(lambda x: x != 0, trgs[index])
        print ('[{:3}] {}'.format('ref', _index2sentence(t_filter, trg_vocab_i2w)))

        trans = trans_onesent(s_filter, network, switchs, mode, trg_vocab_i2w, k, kl, lm, tvcb, ptv)
        print (trans)
        trans_bfilter = _index2sentence(trans, trg_vocab_i2w, ptv)
        print ('[{:3}] {}\n'.format('bef', trans_bfilter))

        trans_filter = _filter_reidx(bos_id, eos_id, trans, switchs[-2], ptv)
        trans_filter = _index2sentence(trans_filter, trg_vocab_i2w, ptv)
        print ('[{:3}] {}\n'.format('out', trans_filter))


@exeTime
def single_trans_valid(x_iter, fs, switchs, trg_vocab_i2w, k=10, mode=0):
    total_trans = []
    for idx, line in enumerate(x_iter):
        s_filter = filter(lambda x: x != 0, line)
        trans = trans_onesent(s_filter, fs, switchs, mode, trg_vocab_i2w, k, kl)
        total_trans.append(trans)
        if numpy.mod(idx + 1, 10) == 0:
            sys.stdout.write('Sample {} Done'.format((idx + 1)))
    sys.stdout.write('Done ...')
    sys.stdout.flush()
    return '\n'.join(total_trans)

from multiprocessing import Process, Queue


def translate(queue, rqueue, pid, network, switchs, mode, trg_vocab_i2w, k, kl, lm=None, tvcb=None):

    while True:
        req = queue.get()
        if req == None:
            break

        idx, src = req[0], req[1]
        print ('{}-{}'.format(pid, idx))
        sys.stdout.flush()
        s_filter = filter(lambda x: x != 0, src)
        trans = trans_onesent(s_filter, network, switchs, mode, trg_vocab_i2w, k, kl, lm, tvcb)

        # print 'trans seq: ', seq
        rqueue.put((idx, trans))

    return


@exeTime
def multi_process(x_iter, network, switchs, mode, trg_vocab_i2w, k=10, n_process=5, kl=0., lm=None,
                  tvcb=None):
    queue = Queue()
    rqueue = Queue()
    processes = [None] * n_process
    for pidx in xrange(n_process):
        processes[pidx] = Process(target=translate, args=(queue, rqueue, pidx, network, switchs, mode,
                                                          trg_vocab_i2w, k, kl, lm, tvcb))
        processes[pidx].start()

    def _send_jobs(x_iter):
        for idx, line in enumerate(x_iter):
            # print idx, line
            queue.put((idx, line))
        return idx + 1

    def _finish_processes():
        for pidx in xrange(n_process):
            queue.put(None)

    def _retrieve_jobs(n_samples):
        trans = [None] * n_samples
        for idx in xrange(n_samples):
            resp = rqueue.get()
            trans[resp[0]] = resp[1]
            if numpy.mod(idx + 1, 1) == 0:
                print ('Sample {}/{} Done'.format((idx + 1), n_samples))
                sys.stdout.flush()
        return trans

    print ('Translating ...')
    n_samples = _send_jobs(x_iter)     # sentence number in source file
    trans_res = _retrieve_jobs(n_samples)
    _finish_processes()
    print ('Done ...')

    return '\n'.join(trans_res)


if __name__ == "__main__":
    import sys
    res = valid_bleu(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    print(res)
