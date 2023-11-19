# -*- coding: UTF-8 -*-

# author: Deqiang Li
# datetime: 2023/8/7 7:42 PM
# software: PyCharm
import os
import argparse
import functools

import numpy as np
import torch

from hashsmooth import JaccardLSHTransformerTorch
from sec_classifiers.drebin.drebin import DrebinSVM, DrebinNN, HashSmooth4Drebin, RandomSmooth4Drebin
from tools import utils
from sec_classifiers.dataset import Dataset
from sec_classifiers.hrat_malscan.randomsmooth_hrat_malscan_det_fs import RandomTransformer
from sec_classifiers.drebin.drebin_adaptive_attack import PGDl1

torch.manual_seed(23456)
torch.cuda.manual_seed(23456)
np.random.seed(23456)

atta_argparse = argparse.ArgumentParser(description='arguments for pgd-l1 attack')

atta_argparse.add_argument('--steps', type=int, default=10,
                           help='maximum number of perturbations.')
# atta_argparse.add_argument('--real', action='store_true', default=False,
#                            help='whether produce the perturbed apks.')
atta_argparse.add_argument('--seed', type=int, default=23456,
                           help='Random seed for reproduction')
atta_argparse.add_argument('--cuda', action='store_true', default=False,
                           help='whether use cuda enable gpu or not.')
atta_argparse.add_argument('--dataset_dir', type=str, default='',
                           help='Folder path to dataset directory.')
atta_argparse.add_argument('--dataset_name', type=str, default='drebin',
                           help='Dataset name.')
atta_argparse.add_argument('--batch_size', type=int, default=64,
                           help='computation upon a batch of data instances for saving RAM')
atta_argparse.add_argument('--sub_k', type=int, default=64,
                           help='Number of hash functions.')
atta_argparse.add_argument('--alpha', type=float, default=0.05,
                           help='Significance level of hypotheses testing.')
atta_argparse.add_argument('--n_sampling', type=int, default=100,
                           help='Number of sampling times for estimating the predictive label.')
atta_argparse.add_argument('--save_path', type=str, default='',
                           help='Folder path to save results.')
atta_argparse.add_argument('--model', type=str, default='drebin_svm',
                           choices=['svm', 'dnn'],
                           help="model type, choose from 'svm' and 'dnn'.\n")
atta_argparse.add_argument('--smooth', type=str, default='none',
                           choices=['none', 'random', 'hash'],
                           help="smooth method, choose from 'random' or 'hash'.\n")

logger = utils.logging.getLogger("pgdl1-attack")
logger.addHandler(utils.ErrorHandler)


def _main():
    args = atta_argparse.parse_args()
    assert args.save_path != '', "Expect a saving path."
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
        input_transfermor = JaccardLSHTransformerTorch(sub_k=args.sub_k,  # initialize this value afterwards
                                                       null_value=0,
                                                       seed=args.seed,
                                                       )
        classifier = HashSmooth4Drebin(classifier, num_of_classes=2,
                                       hash_methods=[input_transfermor],
                                       n_subfeatures=[input_dim],
                                       k_hashcode=args.sub_k,
                                       max_k=args.sub_k,
                                       max_radii=[],
                                       n_grids=[],
                                       default_mode=True,
                                       model_save_dir=os.path.join(args.save_path,
                                                                   'hash_{}_model'.format(args.model))
                                       )
        classifier.predict = functools.partial(classifier.predict, n_sampling=args.n_sampling, alpha=args.alpha)
        classifier.get_loss = functools.partial(classifier.get_loss, n=args.n_sampling, batch_size=args.batch_size)
    elif args.smooth == 'random':
        input_transfermor = RandomTransformer(k_randomcode=args.sub_k, reuse_noise=True, seed=args.seed)
        classifier = RandomSmooth4Drebin(classifier, num_of_classes=2, transform_method=input_transfermor,
                                         max_k=args.sub_k,
                                         default_mode=True,
                                         model_save_dir=os.path.join(args.save_path,
                                                                     'random_{}_model'.format(args.model)))
        classifier.predict = functools.partial(classifier.predict, n_sampling=args.n_sampling, alpha=args.alpha)
        classifier.get_loss = functools.partial(classifier.get_loss, n=args.n_sampling,
                                                k_per_instance=args.sub_k, batch_size=args.batch_size)
    else:
        pass

    # test
    y_prediction = []
    classifier.load_model()
    for idx, (test_x_batch, test_y_batch) in enumerate(test_mal_producer):
        test_x_batch = test_x_batch.to(device)
        y_pred = classifier.predict(test_x_batch).cpu().numpy()
        y_prediction.append(y_pred)
    y_prediction = np.concatenate(y_prediction)
    assert len(y_prediction) == len(mal_test_y)
    if hasattr(classifier, 'ABSTAIN'):
        abstain_flag = y_prediction == classifier.ABSTAIN
        abstain_ratio = np.sum(abstain_flag) / float(len(y_prediction))
        print('Abstain ratio: {}.'.format(abstain_ratio))
        accuracy = (mal_test_y[~abstain_flag] == y_prediction[~abstain_flag]).sum() / float(len(y_prediction)) + abstain_ratio
    else:
        accuracy = (mal_test_y == y_prediction).sum() / float(len(y_prediction))
    logger.info("Model of {}_{} achieves the accuracy on malware test dataset: {:.4f}%".format(args.model, args.smooth + str(args.sub_k),
                                                                                               accuracy * 100))

    # attack
    constraints = np.load(os.path.join(dataset.dataset_path, 'constraints.npz'))
    pgdl1 = PGDl1(constraints['insertion'], constraints['removal'], None, None, is_attacker=True, device=device)
    adv, adv_prediction = [], []
    for idx, (test_x_batch, test_y_batch) in enumerate(test_mal_producer):
        test_x_batch, test_y_batch = test_x_batch.to(device), test_y_batch.to(device)
        adv_x = pgdl1.perturb(classifier, test_x_batch, test_y_batch, steps=args.steps, verbose=True)
        # print(torch.sum(torch.abs(adv_x - test_x_batch), dim=-1))
        adv_y_pred = classifier.predict(adv_x).cpu().numpy()
        adv_prediction.append(adv_y_pred)
        adv.append(adv_x.cpu().numpy())
    adv = np.vstack(adv)
    adv_prediction = np.concatenate(adv_prediction)

    l1_distance = np.sum(np.abs(adv - mal_test_x), axis=-1)
    print("dubug num: ", np.sum(l1_distance == 0.))

    if hasattr(classifier, 'ABSTAIN'):
        abstain_flag = adv_prediction == classifier.ABSTAIN
    else:
        abstain_flag = np.array([False] * len(mal_test_y))
    # filter out the abstain elements
    if np.all(abstain_flag):
        logger.warning("All prediction is abstained.\n")
    else:
        abstain_ratio = np.sum(abstain_flag) / float(len(adv_prediction))
        print('Abstain ratio: {}.'.format(abstain_ratio))
        adv_accuracy = (mal_test_y[~abstain_flag] == adv_prediction[~abstain_flag]).sum() / float(
            len(adv_prediction)) + abstain_ratio
        # adv_accuracy = (mal_test_y[~abstain_flag] == adv_prediction[~abstain_flag]).sum() / np.sum(~abstain_flag)
        logger.info('Abstain ratio: {}.'.format(abstain_ratio))
        logger.info(
            "Model of {}_{} incorported with {} achieves the accuracy {:.4f}% under adversarial attack with {} steps.".format(
                args.model,
                args.sub_k,
                args.smooth,
                adv_accuracy * 100,
                args.steps
                ))
    return
    # save
    if not os.path.exists(os.path.join(dataset.dataset_path, 'adv-examples')):
        utils.mkdir(os.path.join(dataset.dataset_path, 'adv-examples'))
    np.savez(os.path.join(dataset.dataset_path, 'adv-examples', '{}_{}_{}_adv.npz'.format(args.model, args.smooth + str(args.sub_k), args.steps)),
             adv=adv, mal_pred=adv_prediction)


if __name__ == "__main__":
    _main()
