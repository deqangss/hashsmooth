from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import argparse

import torch
import random
import numpy as np
import functools
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
from mimicry import Mimicry

atta_argparse = argparse.ArgumentParser(description='arguments for mimicry attack')
atta_argparse.add_argument('--trials', type=int, default=10,
                           help='number of benign samples for perturbing one malicious file.')
atta_argparse.add_argument('--n_ben', type=int, default=5000,
                           help='number of benign samples.')

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
atta_argparse.add_argument('--save_path', type=str, default='',
                           help='Folder path to save results.')
atta_argparse.add_argument('--model', type=str, default='drebin_svm',
                           choices=['svm', 'dnn'],
                           help="model type, choose from 'svm' and 'dnn'.\n")
atta_argparse.add_argument('--smooth', type=str, default='none',
                           choices=['none', 'random', 'sparsity', 'hash'],
                           help="smooth method, choose from 'random' or 'hash'.\n")
atta_argparse.add_argument('--model_name', type=str, default='xxxxxxxx-xxxxxx', help='model timestamp.')


logger = drebin_utils.logging.getLogger("mimicry-attack")
logger.addHandler(drebin_utils.ErrorHandler)

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def _main():
    args = atta_argparse.parse_args()
    assert args.save_path != '', "Expect a saving path."
    if not os.path.exists(args.save_path):
        drebin_utils.mkdir(args.save_path)

    if args.cuda:
        assert torch.cuda.is_available(), "No GPU device."
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    set_seed(args.seed)

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

    mal_count = len(mal_test_y)
    ben_count = len(ben_test_x)
    if mal_count <= 0 and ben_count <= 0:
        return
    test_mal_producer = dataset.get_dataloader(*(mal_test_x, mal_test_y))
    test_ben_producer = dataset.get_dataloader(*(ben_test_x, ben_test_y))
    # test
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



    logger.info("Load model parameters from {}.".format(classifier.model_save_path))
    classifier.eval()
    ben_feature_vectors = []
    with torch.no_grad():
        c = args.n_ben if args.n_ben < ben_count else ben_count
        for ben_x, ben_y in test_ben_producer:
            ben_x = ben_x.to(device)
            ben_feature_vectors.append(ben_x)
            if len(ben_feature_vectors) * args.batch_size >= c:
                break
        ben_feature_vectors = torch.vstack(ben_feature_vectors)[:c]

    constraints = np.load(os.path.join(dataset.dataset_path, 'constraints.npz'))
    attack = Mimicry(ben_feature_vectors, constraints['insertion'], constraints['removal'], device=device)
    success_flag_list = []
    x_mod_list, adv_prediction = [], []
    for idx, (test_x_batch, test_y_batch) in enumerate(test_mal_producer):
        test_x_batch, test_y_batch = test_x_batch.to(device), test_y_batch.to(device)
        _flag, x_mod = attack.perturb(classifier,
                                      test_x_batch,
                                      test_y_batch,
                                      trials=args.trials,
                                      valid_dim=10000,
                                      seed=args.seed)
        success_flag_list.append(_flag)
        logger.info(
            f"The attack effectiveness under mimicry attack is {np.sum(_flag) / float(len(_flag)) * 100}%.")
        x_mod_list.append(x_mod)
        adv_y_pred = classifier.predict(x_mod).cpu().numpy()
        print(adv_y_pred)
        adv_prediction.append(adv_y_pred)
    success_flag = np.concatenate(success_flag_list)
    logger.info(f"The mean accuracy on perturbed malware is {(1. - np.sum(success_flag) / float(mal_count)) * 100}%.")

    adv = torch.vstack(x_mod_list).cpu().numpy()
    adv_prediction = np.concatenate(adv_prediction)

    if hasattr(classifier, 'ABSTAIN'):
        abstain_flag = adv_prediction == classifier.ABSTAIN
    else:
        abstain_flag = np.array([False] * len(mal_test_y))
    # filter out the abstain elements
    if np.all(abstain_flag):
        logger.warning("All prediction is abstained.\n")
    else:
        abstain_ratio = np.sum(abstain_flag) / float(len(adv_prediction))
        adv_accuracy = (mal_test_y[~abstain_flag] == adv_prediction[~abstain_flag]).sum() / float(
            len(adv_prediction)) + abstain_ratio
        # adv_accuracy = (mal_test_y[~abstain_flag] == adv_prediction[~abstain_flag]).sum() / np.sum(~abstain_flag)
        logger.info('Abstain ratio: {}.'.format(abstain_ratio))
        logger.info(
            "Model of {}_{} incorported with {} achieves the accuracy {:.4f}% under adversarial attack.".format(
                args.model,
                args.K,
                args.smooth,
                adv_accuracy * 100
            ))

    # save
    if not os.path.exists(os.path.join(os.path.dirname(classifier.model_save_path), 'adv-examples')):
        drebin_utils.mkdir(os.path.join(os.path.dirname(classifier.model_save_path), 'adv-examples'))
    np.savez(os.path.join(os.path.dirname(classifier.model_save_path), 'adv-examples',
                          '{}_{}_mimicry_adv.npz'.format(args.model, args.smooth + str(args.K))),
             adv=adv, mal_pred=adv_prediction)




if __name__ == '__main__':
    _main()
