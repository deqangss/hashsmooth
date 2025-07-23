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
from hashsmooth import JaccardLSHTransformer, JaccardLSHTransformerTorch, HammingLSHTransformerTorch
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
                    choices=['none', 'random', 'sparsity', 'hash'],
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
    train_x_y, val_x_y, test_x_y = dataset.load()
    input_dim = train_x_y[0].shape[1]
    # if args.smooth == 'hash':
    #     train_x= dataset.preprocess_hash_dummy_feature(train_x_y[0])
    #     val_x = dataset.preprocess_hash_dummy_feature(val_x_y[0])
    #     test_x = dataset.preprocess_hash_dummy_feature(test_x_y[0])
    #     train_x_y, val_x_y, test_x_y = (train_x, train_x_y[1]), (val_x, val_x_y[1]), (test_x, test_x_y[1])
    train_x_y_producer = dataset.get_dataloader(*train_x_y)
    val_x_y_producer = dataset.get_dataloader(*val_x_y)
    test_x_y_producer = dataset.get_dataloader(*test_x_y)
    if args.model == 'svm':
        classifier = DrebinSVM(input_dim, 1, args.batch_size, os.path.join(args.save_path, 'svm_model'))
        classifier.model.to(device)
        train_model = classifier.fit
        predict = classifier.predict
    elif args.model == 'dnn':
        classifier = DrebinNN(input_dim, 2, args.batch_size, os.path.join(args.save_path, 'dnn_model'))
        classifier.model.to(device)
        train_model = classifier.fit
        predict = classifier.predict
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
        train_model = functools.partial(classifier.fit, n_sampling=args.n_sampling)
        predict = functools.partial(classifier.predict, n_sampling=args.n_sampling, alpha=args.alpha)
    elif args.smooth == 'sparsity':
        classifier = SparsitySmooth4Drebin(classifier, num_of_classes=2,
                                         pf_minus=args.pf_minus,
                                         pf_plus=args.pf_plus,
                                         default_mode=True,
                                         model_save_dir=os.path.join(args.save_path,
                                                                     'sparsity_{}_model'.format(args.model)))
        train_model = functools.partial(classifier.fit, n_sampling=args.n_sampling)
        predict = functools.partial(classifier.predict, n_sampling=args.n_sampling, alpha=args.alpha)
    elif args.smooth == 'hash':
        # input_transformer = JaccardLSHTransformerTorch(sub_k=args.K,  # initialize this value afterwards
        #                                                null_value=0.0,
        #                                                seed=args.seed,
        #                                                )
        input_transformer = HammingLSHTransformerTorch(dimension=input_dim,
                                                       sub_k=args.K,
                                                       null_value=0.0,
                                                       seed=args.seed,
                                                       device=device
                                                       )

        classifier = HashSmooth4Drebin(classifier, num_of_classes=2,
                                       hash_method=input_transformer,
                                       max_k=args.K,
                                       input_dim=input_dim,
                                       default_mode=True,
                                       model_save_dir=os.path.join(args.save_path,
                                                                   'hash_{}_model'.format(args.model))
                                       )
        train_model = functools.partial(classifier.fit, n_sampling=args.n_sampling)
        predict = functools.partial(classifier.predict, n_sampling=args.n_sampling, alpha=args.alpha)
    else:
        raise ValueError

    # train
    train_model(train_x_y_producer, val_x_y_producer, epochs=args.epochs, learning_rate=args.lr, device=device, verbose=True)

    # test
    y_prediction = []
    classifier.load_model()
    for idx, (test_x_batch, test_y_batch) in enumerate(test_x_y_producer):
        test_x_batch = test_x_batch.to(device)
        y_pred = predict(test_x_batch).cpu().numpy()
        y_prediction.append(y_pred)
    y_prediction = np.concatenate(y_prediction)
    assert len(y_prediction) == len(test_x_y[1])

    outlier_indicator = y_prediction[:] == -1
    y_true = test_x_y[1][~outlier_indicator]
    y_pred = y_prediction[~outlier_indicator]
    accuracy, b_accuracy, fnr, fpr, f1, recall_score = measurement(y_true, y_pred)
    logger.info('Filter out outlier: {}.'.format(np.sum(outlier_indicator)))
    logger.info(
        "Model of {} achieves the accuracy: {:.4f}%, balanced accuracy: {:.4f}%".format(args.model, accuracy * 100,
                                                                                        b_accuracy * 100))
    MSG = "False Negative Rate (FNR) is {:.5f}%, False Positive Rate (FPR) is {:.5f}%, F1 score is {:.5f}%, recall score is {:.5f}%"
    logger.info(MSG.format(fnr * 100, fpr * 100, f1 * 100, recall_score * 100))

    y_prediction[outlier_indicator] = 1  # treat as malware
    accuracy, b_accuracy, fnr, fpr, f1, recall_score = measurement(test_x_y[1], y_prediction)
    logger.info('Set outlier as malware prediction:')
    logger.info(
        "Model of {} achieves the accuracy: {:.4f}%, balanced accuracy: {:.4f}%".format(args.model, accuracy * 100,
                                                                                        b_accuracy * 100))
    MSG = "False Negative Rate (FNR) is {:.5f}%, False Positive Rate (FPR) is {:.5f}%, F1 score is {:.5f}%, recall score is {:.5f}%"
    logger.info(MSG.format(fnr * 100, fpr * 100, f1 * 100, recall_score * 100))


def measurement(_y_true, _y_pred):
    from sklearn.metrics import f1_score, accuracy_score, recall_score, confusion_matrix, balanced_accuracy_score
    accuracy = accuracy_score(_y_true, _y_pred)
    b_accuracy = balanced_accuracy_score(_y_true, _y_pred)
    tn, fp, fn, tp = confusion_matrix(_y_true, _y_pred).ravel()
    fpr = fp / float(tn + fp)
    fnr = fn / float(tp + fn)
    f1 = f1_score(_y_true, _y_pred, average='macro')

    recall_score = recall_score(_y_true, _y_pred)
    return accuracy, b_accuracy, fnr, fpr, f1, recall_score



if __name__ == "__main__":
    _main()