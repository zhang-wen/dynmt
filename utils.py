# -*- coding: utf-8 -*-

import numpy
import logging
from itertools import izip
import time
import os
import shutil

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# --exeTime


def exeTime(func):
    def newFunc(*args, **args2):
        t0 = time.time()
        print "@%s, {%s} start" % (time.strftime("%X", time.localtime()), func.__name__)
        back = func(*args, **args2)
        print "@%s, {%s} end" % (time.strftime("%X", time.localtime()), func.__name__)
        print "@%.3fs taken for {%s}" % (time.time() - t0, func.__name__)
        return back
    return newFunc


def init_dirs(models_out_dir, val_out_dir, test_out_dir, **kwargs):
    if os.path.exists(models_out_dir):
        shutil.rmtree(models_out_dir)
        logger.info('{} exists'.format(models_out_dir))
    else:
        os.mkdir(models_out_dir)
        logger.info('create {}'.format(models_out_dir))

    if not val_out_dir == '':
        if os.path.exists(val_out_dir):
            shutil.rmtree(val_out_dir)
            logger.info('{} exists'.format(val_out_dir))
        else:
            os.mkdir(val_out_dir)
            logger.info('create {}'.format(val_out_dir))

    if not test_out_dir == '':
        if os.path.exists(test_out_dir):
            shutil.rmtree(test_out_dir)
            logger.info('{} exists'.format(test_out_dir))
        else:
            os.mkdir(test_out_dir)
            logger.info('create {}'.format(test_out_dir))
