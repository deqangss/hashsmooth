# -*- coding: UTF-8 -*-

# author: Deqiang Li
# datetime: 2023/8/7 7:42 PM
# software: PyCharm
import os
import random
import argparse
import functools

import numpy as np
import torch

from hashsmooth import JaccardLSHTransformerTorch
from sec_classifiers.drebin.drebin import DrebinSVM, DrebinNN, HashSmooth4Drebin, RandomSmooth4Drebin
from tools import utils
from sec_classifiers.dataset import Dataset
from sec_classifiers.hrat_malscan.randomsmooth_hrat_malscan_det_fs import RandomTransformer
from sec_classifiers.drebin.drebin_blackbox_attack import EABlackBoxEvasionProblem, EvolutionAA

torch.manual_seed(23456)
torch.cuda.manual_seed(23456)
np.random.seed(23456)

atta_argparse = argparse.ArgumentParser(description='arguments for evolution attack')

atta_argparse.add_argument('--iterations', type=int, default=100,
                           help='number of iterations.')
atta_argparse.add_argument('--n_repetition', type=int, default=2,
                           help='repeat the evolution algorithm (EA) attack.')
atta_argparse.add_argument('--population_size', type=int, default=1000,
                           help='population size.')
atta_argparse.add_argument('--penalty', type=float, default=0.001,
                           help='penalty factor for l1 regularization.')
atta_argparse.add_argument('--stagnation', type=int, default=8,
                           help='terminate the EA when number of the same result occurance.')
atta_argparse.add_argument('--benign_seed_num', type=int, default=30,
                           help='number of benign samples to initialize the starting point.')
atta_argparse.add_argument('--cx_prob', type=float, default=0.5,
                           help='cross-over probability in EA.')
atta_argparse.add_argument('--mut_prob', type=float, default=0.3,
                           help='mutation probability in EA.')
atta_argparse.add_argument('--flip_prob', type=float, default=0.01,
                           help='mutation probability for an individual.')
atta_argparse.add_argument('--tour_selection_k', type=int, default=100,
                           help='number of selected individuals for produce offspring.')
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
atta_argparse.add_argument('--save_path', type=str, default='./results',
                           help='Folder path to save results.')
atta_argparse.add_argument('--model', type=str, default='svm',
                           choices=['svm', 'dnn'],
                           help="model type, choose from 'svm' and 'dnn'.\n")
atta_argparse.add_argument('--smooth', type=str, default='none',
                           choices=['none', 'random', 'hash'],
                           help="smooth method, choose from 'random' or 'hash'.\n")

logger = utils.logging.getLogger("ea-attack")
logger.addHandler(utils.ErrorHandler)


def _main():
    args = atta_argparse.parse_args()
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
    ben_test_y = test_y[test_y == 0]
    ben_test_x = test_x[test_y == 0]
    test_indices = np.arange(len(mal_test_x))
    np.random.seed(args.seed)
    np.random.shuffle(test_indices)
    test_mal_producer = dataset.get_dataloader(*(mal_test_x[test_indices[:200]], mal_test_y[test_indices[:200]]))
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
        classifier.get_confidence = functools.partial(classifier.get_confidence, n_sampling=args.n_sampling)
        name_suffix = args.sub_k
    elif args.smooth == 'random':
        input_transfermor = RandomTransformer(k_randomcode=args.sub_k, reuse_noise=True, seed=args.seed)
        classifier = RandomSmooth4Drebin(classifier, num_of_classes=2, transform_method=input_transfermor,
                                         max_k=args.sub_k,
                                         default_mode=True,
                                         model_save_dir=os.path.join(args.save_path,
                                                                     'random_{}_model'.format(args.model)))
        classifier.predict = functools.partial(classifier.predict, n_sampling=args.n_sampling, alpha=args.alpha)
        classifier.get_confidence = functools.partial(classifier.get_confidence, n_sampling=args.n_sampling,
                                                      k_per_instance=args.sub_k)
        name_suffix = args.sub_k
    else:
        name_suffix = ''
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
    accuracy = (mal_test_y == y_prediction).sum() / float(len(y_prediction))
    logger.info("Model of {} achieves the accuracy on malware test dataset: {:.4f}%".format(args.model, accuracy * 100))

    # attack
    constraints = np.load(os.path.join(dataset.dataset_path, 'constraints.npz'))
    no_of_seed = args.benign_seed_num if args.benign_seed_num < ben_test_x.shape[0] else ben_test_x.shape[0]
    ben_idx = np.random.choice(range(ben_test_x.shape[0]), no_of_seed, replace=False)
    blackbox_attack_problem = EABlackBoxEvasionProblem(classifier,
                                                       input_dim=input_dim,
                                                       population_size=args.population_size,
                                                       penalty_regularizer=args.penalty,
                                                       constraints=(constraints['insertion'], constraints['removal']),
                                                       iterations=args.iterations,
                                                       stagnation=args.stagnation,
                                                       benign_samples_as_seed=ben_test_x[ben_idx],
                                                       seed=args.seed,
                                                       device=device
                                                       )
    ea_attack = EvolutionAA(blackbox_attack_problem,
                            cx_prob=args.cx_prob,
                            mut_prob=args.mut_prob,
                            individual_flip_prob=args.flip_prob,
                            tour_selection_k=args.tour_selection_k,
                            n_repetition=args.n_repetition)
    advs, adv_prediction = [], []
    for idx, (mal_x, mal_y) in enumerate(zip(mal_test_x, mal_test_y)):
        adv_x = ea_attack.perturb(mal_x, mal_y, verbose=True)
        adv_y_pred = classifier.predict(torch.from_numpy(adv_x[None, ...]).to(device).float()).cpu().numpy()
        adv_prediction.append(adv_y_pred)
        advs.append(adv_x)
        print(np.sum(np.abs(adv_x - mal_x), axis=-1), adv_y_pred)
    advs = np.vstack(advs)
    adv_prediction = np.concatenate(adv_prediction)
    adv_accuracy = (mal_test_y == adv_prediction).sum() / float(len(adv_prediction))
    logger.info(
        "Model of {} achieves the accuracy on adversarial test dataset: {:.4f}%".format(args.model, adv_accuracy * 100))
    # save
    if not os.path.exists(os.path.join(dataset.dataset_path, 'adv-examples-ea')):
        utils.mkdir(os.path.join(dataset.dataset_path, 'adv-examples-ea'))
    np.savez(os.path.join(dataset.dataset_path, 'adv-examples-ea',
                          '{}_{}_{}_adv.npz'.format(args.model, name_suffix, args.smooth)),
             adv=advs, mal_pred=adv_prediction, test_idx=test_indices[:200])


if __name__ == "__main__":
    _main()
