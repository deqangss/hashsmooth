from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import time
import logging
import warnings
import sys
import fileinput
import shutil

import hashlib
import random
import string
import base64
import re

try:
    basestring
except Exception:
    basestring = str

logging.basicConfig(level=logging.INFO,
                    filename=os.path.join(os.getcwd(), time.strftime("%Y%m%d-%H%M%S") + ".log"),
                    filemode="w",
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S')
ErrorHandler = logging.StreamHandler()
ErrorHandler.setFormatter(logging.Formatter('%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s'))


def dump_pickle(data, path):
    try:
        import pickle as pkl
    except Exception as e:
        import cPickle as pkl

    if not os.path.exists(os.path.dirname(path)):
        mkdir(os.path.dirname(path))
    with open(path, 'wb') as wr:
        pkl.dump(data, wr)
    return True


def read_pickle(path):
    try:
        import pickle as pkl
    except Exception as e:
        import cPickle as pkl

    if os.path.isfile(path):
        with open(path, 'rb') as fr:
            return pkl.load(fr)
    else:
        raise IOError("The {0} is not been found.".format(path))


def dump_joblib(data, path):
    if not os.path.exists(os.path.dirname(path)):
        mkdir(os.path.dirname(path))

    try:
        from sklearn.externals import joblib
        with open(path, 'wb') as wr:
            joblib.dump(data, wr)
    except IOError:
        raise IOError("Dump data failed.")


def read_joblib(path):
    from sklearn.externals import joblib
    if os.path.isfile(path):
        with open(path, 'rb') as fr:
            return joblib.load(fr)
    else:
        raise IOError("The {0} is not a file.".format(path))


def read_txt(path, mode='r'):
    if os.path.isfile(path):
        with open(path, mode) as f_r:
            lines = f_r.read().strip().splitlines()
            return lines
    else:
        raise ValueError("{} does not seen like a file path.\n".format(path))


def dump_txt(data_str, path, mode='w'):
    if not isinstance(data_str, str):
        raise TypeError

    with open(path, mode) as f_w:
        f_w.write(data_str)


def mkdir(target):
    try:
        if os.path.isfile(target):
            target = os.path.dirname(target)

        if not os.path.exists(target):
            os.makedirs(target)
        return 0
    except IOError as e:
        sys.stderr.write(e)
        sys.exit(1)


def guess_file_name(folder, name):
    assert os.path.isdir(folder)
    if not os.path.exists(folder):
        raise FileExistsError("No such {}.\n".format(folder))
    file_names = os.listdir(folder)
    rtn_name = [n for n in file_names if name in n]
    return rtn_name
