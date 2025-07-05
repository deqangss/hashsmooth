# -*- coding: UTF-8 -*-

# author: Deqiang Li
# datetime: 2023/8/7 7:42 PM
# software: PyCharm
import os
import sys
import random
import argparse
import functools

import numpy as np
import torch
import drebin_utils
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
from drebin_blackbox_attack import EABlackBoxEvasionProblem, EvolutionAA

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
atta_argparse.add_argument('--K', type=int, default=64,
                           help='Number of hash functions.')
atta_argparse.add_argument('--pf_minus', type=float, default=0.9,
                    help='The probability to flip a one to a zero.')
atta_argparse.add_argument('--pf_plus', type=float, default=0.1,
                    help='The probability to flip a zero to a one')
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
                           choices=['none', 'random', 'sparsity', 'hash'],
                           help="smooth method, choose from 'random' or 'hash'.\n")

logger = drebin_utils.logging.getLogger("ea-attack")
logger.addHandler(drebin_utils.ErrorHandler)


def _main():
    args = atta_argparse.parse_args()

    if args.cuda:
        assert torch.cuda.is_available(), "No GPU device."
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    # obtain data
    dataset = Dataset(args.dataset_dir, args.dataset_name, args.batch_size)
    _1, _2, test_x_y = dataset.load()
    test_x, test_y = test_x_y
    input_dim = test_x.shape[1]
    if args.smooth == 'hash':
        test_x = dataset.preprocess_hash_dummy_feature(test_x)
    mal_test_y = test_y[test_y == 1]
    mal_test_x = test_x[test_y == 1]
    ben_test_y = test_y[test_y == 0]
    ben_test_x = test_x[test_y == 0]

    test_indices = np.arange(len(mal_test_x))
    np.random.seed(args.seed)
    np.random.shuffle(test_indices)
    mal_test_x_sel = mal_test_x[test_indices[:100]] # todo: check the number of examples
    mal_test_y_sel = mal_test_y[test_indices[:100]]

    test_mal_producer = dataset.get_dataloader(*(mal_test_x_sel, mal_test_y_sel))
    if args.model == 'svm':
        classifier = DrebinSVM(input_dim, 1, args.batch_size, os.path.join(args.save_path, 'svm_model'))
        classifier.model.to(device)
    elif args.model == 'dnn':
        classifier = DrebinNN(input_dim, 2, args.batch_size, os.path.join(args.save_path, 'dnn_model'))
        classifier.model.to(device)
    else:
        raise ValueError("Choose either 'drebin_svm' or 'drebin_nn'.\n")

    if args.smooth == 'none':
        pass
    elif args.smooth == 'random':
        input_transformer = RandomTransformer(k_randomcode=args.K, mask_id=0.0, reuse_noise=True, seed=args.seed)
        classifier = RandomSmooth4Drebin(classifier, num_of_classes=2, transform_method=input_transformer,
                                         max_k=args.K,
                                         default_mode=True,
                                         model_save_dir=os.path.join(args.save_path,
                                                                     'random_{}_model'.format(args.model)))
        classifier.predict = functools.partial(classifier.predict, n_sampling=args.n_sampling, alpha=args.alpha)
        classifier.get_loss = functools.partial(classifier.get_loss, n=args.n_sampling, batch_size=args.batch_size)
    elif args.smooth == 'sparsity':
        classifier = SparsitySmooth4Drebin(classifier, num_of_classes=2,
                                           pf_minus=args.pf_minus,
                                           pf_plus=args.pf_plus,
                                           default_mode=True,
                                           model_save_dir=os.path.join(args.save_path,
                                                                       'sparsity_{}_model'.format(args.model)))
        classifier.predict = functools.partial(classifier.predict, n_sampling=args.n_sampling, alpha=args.alpha)
        classifier.get_loss = functools.partial(classifier.get_loss, n=args.n_sampling, batch_size=args.batch_size)
    elif args.smooth == 'hash':
        input_transformer = JaccardLSHTransformerTorch(sub_k=args.K,  # initialize this value afterwards
                                                       null_value=0.0,
                                                       seed=args.seed,
                                                       )
        classifier = HashSmooth4Drebin(classifier, num_of_classes=2,
                                       hash_method=input_transformer,
                                       max_k=args.K,
                                       input_dim=input_dim,
                                       default_mode=True,
                                       model_save_dir=os.path.join(args.save_path,
                                                                   'hash_{}_model'.format(args.model))
                                       )
        classifier.predict = functools.partial(classifier.predict, n_sampling=args.n_sampling, alpha=args.alpha)
        classifier.get_loss = functools.partial(classifier.get_loss, n=args.n_sampling, batch_size=args.batch_size)
    else:
        raise ValueError

    # test
    y_prediction = []
    classifier.load_model()
    for idx, (test_x_batch, test_y_batch) in enumerate(test_mal_producer):
        test_x_batch = test_x_batch.to(device)
        y_pred = classifier.predict(test_x_batch).cpu().numpy()
        y_prediction.append(y_pred)
    y_prediction = np.concatenate(y_prediction)
    assert len(y_prediction) == len(mal_test_y_sel)
    if hasattr(classifier, 'ABSTAIN'):
        abstain_flag = y_prediction == classifier.ABSTAIN
        abstain_ratio = np.sum(abstain_flag) / float(len(y_prediction))
        logger.info('Abstain ratio: {}.'.format(abstain_ratio))
        accuracy = (mal_test_y_sel[~abstain_flag] == y_prediction[~abstain_flag]).sum() / float(len(y_prediction)) + abstain_ratio
    else:
        accuracy = (mal_test_y_sel == y_prediction).sum() / float(len(y_prediction))
    logger.info("Model of {}_{} achieves the accuracy on malware test dataset: {:.4f}%".format(args.model, args.smooth + str(args.K),
                                                                                               accuracy * 100))

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
    for idx, (mal_x, mal_y) in enumerate(zip(mal_test_x_sel, mal_test_y_sel)):
        adv_x = ea_attack.perturb(mal_x, mal_y, verbose=True)
        adv_y_pred = classifier.predict(torch.from_numpy(adv_x[None, ...]).to(device).float()).cpu().numpy()
        print("outer adv pred: ", adv_y_pred)
        adv_prediction.append(adv_y_pred)
        advs.append(adv_x)
        print(np.sum(np.abs(adv_x - mal_x), axis=-1), adv_y_pred)
    advs = np.vstack(advs)
    adv_prediction = np.concatenate(adv_prediction)
    if hasattr(classifier, 'ABSTAIN'):
        abstain_flag = adv_prediction == classifier.ABSTAIN
    else:
        abstain_flag = np.array([False] * len(mal_test_y_sel))
        # filter out the abstain elements
    if np.all(abstain_flag):
        logger.warning("All prediction is abstained.\n")
    else:
        abstain_ratio = np.sum(abstain_flag) / float(len(adv_prediction))
        adv_accuracy = (mal_test_y_sel[~abstain_flag] == adv_prediction[~abstain_flag]).sum() / float(
            len(adv_prediction)) + abstain_ratio
        # adv_accuracy = (mal_test_y[~abstain_flag] == adv_prediction[~abstain_flag]).sum() / np.sum(~abstain_flag)
        logger.info('Abstain ratio: {}.'.format(abstain_ratio))
        logger.info(
            "Model of {}_{} incorported with {} achieves the accuracy {:.4f}% under adversarial attack.".format(
                args.model,
                args.K,
                args.smooth,
                adv_accuracy * 100,
            ))

    # save
    if not os.path.exists(os.path.join(os.path.dirname(classifier.model_save_path), 'adv-examples-ea')):
        drebin_utils.mkdir(os.path.join(os.path.dirname(classifier.model_save_path), 'adv-examples-ea'))
    np.savez(os.path.join(os.path.dirname(classifier.model_save_path), 'adv-examples-ea',
                          '{}_{}_{}_adv.npz'.format(args.model, args.smooth, args.K)),
             adv=advs, mal_pred=adv_prediction, test_idx=test_indices[:200])


if __name__ == "__main__":
    _main()
