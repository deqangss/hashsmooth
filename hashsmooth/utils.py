from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import signal
import time
import warnings
import sys
import fileinput
import shutil
import pickle as pkl

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
    if not os.path.exists(os.path.dirname(path)):
        mkdir(os.path.dirname(path))
    with open(path, 'wb') as wr:
        pkl.dump(data, wr)
    return True


def read_pickle(path):
    if os.path.isfile(path):
        with open(path, 'rb') as fr:
            return pkl.load(fr)
    else:
        raise IOError("The {0} is not been found.".format(path))


def dump_joblib(data, path):
    if not os.path.exists(os.path.dirname(path)):
        mkdir(os.path.dirname(path))

    try:
        with open(path, 'wb') as wr:
            joblib.dump(data, wr)
    except IOError:
        raise IOError("Dump data failed.")


def read_joblib(path):
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

def pool_initializer():
    """Ignore CTRL+C in the worker process."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


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


def retrive_files_set(base_dir, dir_ext, file_ext):
    """
    get file paths given the directory
    :param base_dir: basic directory
    :param dir_ext: directory append at the rear of base_dir
    :param file_ext: file extension
    :return: set of file paths. Avoid the repetition
    """
    def get_file_name(root_dir, file_ext):

        for dir_path, dir_names, file_names in os.walk(root_dir):
            for file_name in file_names:
                _ext = file_ext
                if os.path.splitext(file_name)[1] == _ext:
                    yield os.path.join(dir_path, file_name)
                elif '.' not in file_ext:
                    _ext = '.' + _ext

                    if os.path.splitext(file_name)[1] == _ext:
                        yield os.path.join(dir_path,file_name)
                    else:
                        pass
                else:
                    pass

    if file_ext is not None:
        file_exts = file_ext.split("|")
    else:
        file_exts = ['']
    file_path_set = set()
    for ext in file_exts:
        file_path_set = file_path_set | set(get_file_name(os.path.join(base_dir, dir_ext), ext))

    return file_path_set
