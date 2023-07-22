from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
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
