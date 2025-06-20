# -*- coding: UTF-8 -*-

# author: Deqiang Li
# datetime: 2024/8/7 7:42 PM
# software: PyCharm
import os
import argparse
import functools

import numpy as np
import torch
import sys

import drebin_utils
torch.manual_seed(23456)
torch.cuda.manual_seed(23456)
np.random.seed(23456)

sys.path.append('../../')
sys.path.append('../../../')
sys.path.append('../../../../')
sys.path.append('../../../python_parser')
sys.path.append('../../../../hashsmooth')
sys.path.append('../../../../randomsmooth')
sys.path.append('../../../../torchware')

from randomsmooth.random_tran import RandomTransformer
from hashsmooth import JaccardLSHTransformer, JaccardLSHTransformerTorch
from model import DrebinNN, DrebinSVM, RandomSmooth4Drebin, HashSmooth4Drebin, SparsitySmooth4Drebin
from dataset import Dataset

cmd_md = argparse.ArgumentParser(description='arguments for drebin model test')
cmd_md.add_argument('--lr', type=float, default=0.001,
                    help='The learning rate of ML models.')
cmd_md.add_argument('--seed', type=int, default=23456,
                    help='Random seed for reproduction')
cmd_md.add_argument('--cuda', action='store_true', default=False,
                    help='whether use cuda enable gpu or not.')
cmd_md.add_argument('--dataset_dir', type=str, default='',
                    help='Folder path to dataset directory.')
cmd_md.add_argument('--dataset_name', type=str, default='drebin',
                    help='Dataset name.')
cmd_md.add_argument('--batch_size', type=int, default=64,
                    help='computation upon a batch of data instances for saving RAM')
cmd_md.add_argument('--epochs', type=int, default=200,
                    help='number of epochs to train a model')
cmd_md.add_argument('--K', type=int, default=64,
                    help='Number of selected entities or hash functions.')
cmd_md.add_argument('--alpha', type=float, default=0.05,
                    help='Significance level of hypotheses testing.')
cmd_md.add_argument('--pf_minus', type=float, default=0.9,
                    help='The probability to flip a one to a zero.')
cmd_md.add_argument('--pf_plus', type=float, default=0.1,
                    help='The probability to flip a zero to a one')
cmd_md.add_argument('--n_sampling', type=int, default=100,
                    help='Number of sampling times for estimating the predictive label.')
cmd_md.add_argument('--save_path', type=str, default='',
                    help='Folder path to save results.')
cmd_md.add_argument('--model', type=str, default='drebin_svm',
                    choices=['svm', 'dnn'],
                    help="model type, choose from 'svm' and 'dnn'.\n")
cmd_md.add_argument('--smooth', type=str, default='none',
                    choices=['random', 'sparsity', 'hash'],
                    help="smooth method, choose from 'random' or 'hash'.\n")

logger = drebin_utils.logging.getLogger("drebin")
logger.addHandler(drebin_utils.ErrorHandler)

def _main():
    args = cmd_md.parse_args()
    logger.info(vars(args))
    if not os.path.exists(args.save_path):
        drebin_utils.mkdir(args.save_path)
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
    if args.smooth == 'hash':
        mal_test_x = dataset.preprocess_hash_dummy_feature(mal_test_x)
    test_mal_producer = dataset.get_dataloader(*(mal_test_x, mal_test_y))
    input_dim = mal_test_x[0].shape[1]
    if args.model == 'svm':
        classifier = DrebinSVM(input_dim, 1, args.batch_size, os.path.join(args.save_path, 'svm_model'))
        classifier.model.to(device)
    elif args.model == 'dnn':
        classifier = DrebinNN(input_dim, 2, args.batch_size, os.path.join(args.save_path, 'dnn_model'))
        classifier.model.to(device)
    else:
        raise ValueError("Choose either 'drebin_svm' or 'drebin_nn'.\n")

    if args.smooth == 'random':
        input_transformer = RandomTransformer(k_randomcode=args.K, mask_id=0.0, reuse_noise=True, seed=args.seed)
        classifier = RandomSmooth4Drebin(classifier, num_of_classes=2, transform_method=input_transformer,
                                         max_k=args.K,
                                         default_mode=True,
                                         model_save_dir=os.path.join(args.save_path,
                                                                     'random_{}_model'.format(args.model)))
        classifier.certify = functools.partial(classifier.certify, n_selection=args.n_sampling,
                                               n_estimation=args.n_estimation,
                                               alpha=args.alpha)
    elif args.smooth == 'sparsity':
        classifier = SparsitySmooth4Drebin(classifier, num_of_classes=2,
                                         pf_minus=args.pf_minus,
                                         pf_plus=args.pf_plus,
                                         default_mode=True,
                                         model_save_dir=os.path.join(args.save_path,
                                                                     'random_{}_model'.format(args.model)))
        classifier.certify = functools.partial(classifier.certify, n_selection=args.n_sampling,
                                               n_estimation=args.n_estimation,
                                               alpha=args.alpha)
    elif args.smooth == 'hash':
        input_transformer = JaccardLSHTransformerTorch(sub_k=args.K,  # initialize this value afterwards
                                                       null_value=0.0,
                                                       seed=args.seed,
                                                       )
        classifier = HashSmooth4Drebin(classifier, num_of_classes=2,
                                       hash_method=input_transformer,
                                       max_k=args.K,
                                       default_mode=True,
                                       model_save_dir=os.path.join(args.save_path,
                                                                   'hash_{}_model'.format(args.model))
                                       )
        classifier.certify = functools.partial(classifier.certify, n_selection=args.n_sampling,
                                               n_estimation=args.n_estimation,
                                               alpha=args.alpha)
    else:
        raise ValueError

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
    if args.smooth == 'random' or args.smooth == 'hash':
        radii_dis = np.concatenate([rad[0] for rad in radii])
    elif args.smooth == 'sparsity':
        radii_dis = np.vstack([[rad[0], rad[1]] for rad in radii])
    else:
        raise ValueError
    assert len(radii_dis) == len(mal_test_y)
    radius_mean = np.mean(radii_dis.clip(min=0.), axis=-1)
    radius_median = np.median(radii_dis, axis=-1)
    logger.info('\n' + '\n'.join(map(str, radii_dis.tolist())))
    logger.info("Model of {} achieves the certification mean {} and median {} with input dim {}.".format(
        args.model, radius_mean, radius_median, input_dim))
    if not os.path.exists(os.path.join(classifier.model_save_path, 'cert')):
        drebin_utils.mkdir(os.path.join(classifier.model_save_path, 'cert'))
    if args.smooth == 'random' or args.smooth == 'hash':
        np.savez(os.path.join(classifier.model_save_path, 'cert',
                              '{}_{}_{}_radii.npz'.format(args.model, args.smooth, args.K)),
                 radii=radii)
    else:
        np.savez(os.path.join(classifier.model_save_path, 'cert',
                              '{}_{}_{}_{}_radii.npz'.format(args.model, args.smooth, args.pf_minus, args.pf_plus)),
                 radii=radii)



if __name__ == "__main__":
    _main()