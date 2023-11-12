# -*- coding: UTF-8 -*-

# author: Deqiang Li
# datetime: 2023/8/7 7:42 PM
# software: PyCharm
import time
import os
import argparse
import functools

import numpy as np
import torch
import math

from hashsmooth import JaccardLSHTransformerTorch
from sec_classifiers.drebin.drebin import DrebinSVM, DrebinNN, HashSmooth4Drebin, RandomSmooth4Drebin
from tools import utils
from sec_classifiers.dataset import Dataset
from sec_classifiers.hrat_malscan.randomsmooth_hrat_malscan_det_fs import RandomTransformer

torch.manual_seed(23456)
torch.cuda.manual_seed(23456)
np.random.seed(23456)

cert_argparse = argparse.ArgumentParser(description='arguments for drebin certification')

cert_argparse.add_argument('--seed', type=int, default=23456,
                           help='Random seed for reproduction')
cert_argparse.add_argument('--cuda', action='store_true', default=False,
                           help='whether use cuda enable gpu or not.')
cert_argparse.add_argument('--dataset_dir', type=str, default='',
                           help='Folder path to dataset directory.')
cert_argparse.add_argument('--dataset_name', type=str, default='drebin',
                           help='Dataset name.')
cert_argparse.add_argument('--batch_size', type=int, default=64,
                           help='computation upon a batch of data instances for saving RAM')
cert_argparse.add_argument('--sub_k', type=int, default=64,
                           help='Number of hash functions.')
cert_argparse.add_argument('--alpha', type=float, default=0.05,
                           help='Significance level of hypotheses testing.')
cert_argparse.add_argument('--n_sampling', type=int, default=100,
                           help='Number of sampling times for estimating the predictive label.')
cert_argparse.add_argument('--n_estimation', type=int, default=10000,
                           help='Number of sampling times for estimating the confidence level.')
cert_argparse.add_argument('--save_path', type=str, default='./results',
                           help='Folder path to save results.')
cert_argparse.add_argument('--model', type=str, default='drebin_svm',
                           choices=['svm', 'dnn'],
                           help="model type, choose from 'svm' and 'dnn'.\n")
cert_argparse.add_argument('--smooth', type=str, default='none',
                           choices=['none', 'random', 'hash'],
                           help="smooth method, choose from 'random' or 'hash'.\n")

logger = utils.logging.getLogger("Certification")
logger.addHandler(utils.ErrorHandler)


def _main():
    args = cert_argparse.parse_args()
    if not os.path.exists(args.save_path):
        utils.mkdir(args.save_path)

    if args.cuda:
        assert torch.cuda.is_available(), "No GPU device."
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    # obtain data
    dataset = Dataset(args.dataset_dir, args.dataset_name, args.batch_size)
    _1, _2, test_x_y = dataset.load()
    test_x, test_y = test_x_y
    mal_test_y = test_y[test_y == 1]
    mal_test_x = test_x[test_y == 1]
    test_mal_producer = dataset.get_dataloader(*(mal_test_x, mal_test_y))
    input_dim = test_x.shape[1]
    if args.model == 'svm':
        classifier = DrebinSVM(input_dim, 1, args.batch_size, os.path.join(args.save_path, 'svm_model'))
        classifier.model.to(device)
    elif args.model == 'dnn':
        classifier = DrebinNN(input_dim, 2, args.batch_size, os.path.join(args.save_path, 'dnn_model'))
        classifier.model.to(device)
    else:
        raise ValueError("Choose either 'drebin_svm' or 'drebin_nn'.\n")

    if args.smooth == 'hash':
        input_transformer = JaccardLSHTransformerTorch(sub_k=args.sub_k,  # initialize this value afterwards
                                                       null_value=0,
                                                       seed=args.seed,
                                                       )
        classifier = HashSmooth4Drebin(classifier, num_of_classes=2,
                                       hash_methods=[input_transformer],
                                       n_subfeatures=[input_dim],
                                       k_hashcode=args.sub_k,
                                       max_k=args.sub_k,
                                       max_radii=[0.5],
                                       n_grids=[1000],
                                       default_mode=True,
                                       model_save_dir=os.path.join(args.save_path,
                                                                   'hash_{}_model'.format(args.model))
                                       )
        classifier.certify = functools.partial(classifier.certify, n_selection=args.n_sampling,
                                               n_estimation=args.n_estimation,
                                               k_per_instance=args.sub_k, alpha=args.alpha)
    elif args.smooth == 'random':
        input_transformer = RandomTransformer(k_randomcode=args.sub_k, reuse_noise=True, seed=args.seed)
        classifier = RandomSmooth4Drebin(classifier, num_of_classes=2, transform_method=input_transformer,
                                         max_k=args.sub_k,
                                         default_mode=True,
                                         model_save_dir=os.path.join(args.save_path,
                                                                     'random_{}_model'.format(args.model)))
        classifier.certify = functools.partial(classifier.certify, n_selection=args.n_sampling,
                                               n_estimation=args.n_estimation,
                                               k_per_instance=args.sub_k, alpha=args.alpha)
    else:
        raise ValueError("Choose either 'random' or 'hash', but got {}.\n".format(args.smooth))

    # certification
    logger.info("sub k {}, number of selection {}, number of estimation {}, confidence level {}.".format(
        args.sub_k,
        args.n_sampling,
        args.n_estimation,
        args.alpha
    ))

    radii = []
    classifier.load_model()
    for idx, (test_x_batch, test_y_batch) in enumerate(test_mal_producer):
        test_x_batch, test_y_batch = test_x_batch.to(device), test_y_batch.to(device)
        radii_batch = classifier.certify(test_x_batch, test_y_batch)
        radii.append(radii_batch)
    radii = np.concatenate(radii)
    assert len(radii) == len(mal_test_y)
    radius_mean = np.mean(radii.clip(min=0.))
    radius_median = np.median(radii)
    logger.info('\n' + '\n'.join(map(str, radii.tolist())))
    logger.info("Model of {} achieves the certification mean {} and median {} with input dim {}.".format(
        args.model, radius_mean, radius_median, input_dim))
    if not os.path.exists(os.path.join(dataset.dataset_path, 'cert')):
        utils.mkdir(os.path.join(dataset.dataset_path, 'cert'))
    np.savez(os.path.join(dataset.dataset_path, 'cert', '{}_{}_{}_radii.npz'.format(args.model, args.smooth, args.sub_k)),
             radii=radii)


if __name__ == "__main__":
    _main()
