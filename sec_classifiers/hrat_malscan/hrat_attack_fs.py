# -*- coding: UTF-8 -*-

# author: Deqiang Li
# datetime: 2023/8/7 7:42 PM
# software: PyCharm
import time
import os
import argparse
from tqdm import tqdm
import multiprocessing
import functools

import numpy as np
import torch

from hashsmooth import WeightedJaccardLSHTransformerTorch
from sec_classifiers.hrat_malscan.hrat_malscan_det import MalScan
from sec_classifiers.hrat_malscan.hashsmooth_hrat_malscan_det_fs import HashSmooth4MalScan
from tools import utils
from sec_classifiers.hrat_malscan.Utils import trans2triple_rw, trans2triple
from sec_classifiers.hrat_malscan import myenv_withconstraints_fs
from sec_classifiers.hrat_malscan.model import DQN
from sec_classifiers.dataset import Dataset
from sec_classifiers.hrat_malscan.randomsmooth_hrat_malscan_det_fs import RandomTransformer, RandomSmooth4MalScan

ACTION_NUM = 4

torch.manual_seed(23456)
torch.cuda.manual_seed(23456)
np.random.seed(23456)

cmd_md = argparse.ArgumentParser(description='arguments for hrat attack')

cmd_md.add_argument('--memory_cap', type=int, default=16,
                    help='The memory capability of DQN.')
cmd_md.add_argument('--lr', type=float, default=0.00001,
                    help='The learning rate of DQN.')
cmd_md.add_argument('--steep', type=float, default=1.,
                    help='The hyper-parameter of surrogate KNN.')
cmd_md.add_argument('--knn_num', type=int, default=10,
                    help='The neighbors of KNN.')
cmd_md.add_argument('--max_node', type=int, default=5000,
                    help='The maximum number of nodes for saving RAM')
cmd_md.add_argument('--seed', type=int, default=23456,
                    help='Random seed for reproduction')
cmd_md.add_argument('--cuda', action='store_true', default=False,
                    help='whether use cuda enable gpu or not.')
cmd_md.add_argument('--dataset_dir', type=str, default='',
                    help='Folder path to dataset directory.')
cmd_md.add_argument('--dataset_name', type=str, default='malscan',
                    help='Dataset name.')
cmd_md.add_argument('--batch_size', type=int, default=64,
                    help='computation upon a batch of data instances for saving RAM')
cmd_md.add_argument('--test', action='store_true', default=False,
                    help='Predict labels for all test data instances.')
cmd_md.add_argument('--is_benign', action='store_true', default=True,
                    help='Just use benign instances to optimize perturbations.')
cmd_md.add_argument('--sub_k', type=int, default=64,
                    help='Number of hash functions.')
cmd_md.add_argument('--alpha', type=float, default=0.05,
                    help='Significance level of hypotheses testing.')
cmd_md.add_argument('--n_sampling', type=int, default=100,
                    help='Number of sampling times for estimating the predictive label.')
cmd_md.add_argument('--save_path', type=str, default='./results',
                    help='Folder path to save results.')
cmd_md.add_argument('--model', type=str, default='malscan',
                    choices=['malscan', 'random_malscan', 'hash_malscan'],
                    help="model type, choose from 'malscan', 'random_malscan', 'hash_malscan'\n")

logger = utils.logging.getLogger("Hrat-attack")
logger.addHandler(utils.ErrorHandler)


def _main():
    args = cmd_md.parse_args()
    if not os.path.exists(args.save_path):
        utils.mkdir(args.save_path)

    if args.cuda:
        assert torch.cuda.is_available(), "No GPU device."
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    # obtain data
    dataset = Dataset(args.dataset_dir, args.dataset_name, args.batch_size)
    feature_saving_path = dataset.dataset_path
    train_x_y_path = os.path.join(feature_saving_path, "train_db.npz")
    val_x_y_path = os.path.join(feature_saving_path, "val_db.npz")
    if not os.path.exists(train_x_y_path):
        train_pkl, val_pkl, _1 = dataset.load()
        train_x, train_y = get_feature_rpst(train_pkl)
        np.savez(train_x_y_path, train_x=train_x, train_y=train_y)
        val_x, val_y = get_feature_rpst(val_pkl)
        np.savez(val_x_y_path, val_x=val_x, val_y=val_y)
    else:
        train_x_y = np.load(train_x_y_path)
        train_x, train_y = train_x_y['train_x'], train_x_y['train_y']
        val_x_y = np.load(val_x_y_path)
        val_x, val_y = val_x_y['val_x'], val_x_y['val_y']

    if args.model == 'malscan':
        train_x_producer = torch.from_numpy(train_x)  # train_x_producer = dataset.get_dataloader(train_x)
        train_y = torch.from_numpy(train_y).to(device)
        input_transfermor = None
        malscan = MalScan(train_x_producer, train_y, args.batch_size)  # basic classifier
        predict_func = malscan.predict
    elif args.model == 'hash_malscan':
        input_transfermor = WeightedJaccardLSHTransformerTorch(number_of_words=train_x.shape[-1],
                                                               sub_k=args.sub_k,  # initialize this value afterwards
                                                               null_value=0,
                                                               seed=args.seed)
        train_x_train = get_rpst_hashtran(train_x, input_transfermor, args.batch_size, device)
        train_x_producer = torch.from_numpy(train_x_train)
        train_y = torch.from_numpy(train_y).to(device)
        malscan = MalScan(train_x_producer, train_y, args.batch_size)
        malscan = HashSmooth4MalScan(malscan, num_of_classes=2,
                                     hash_methods=[input_transfermor],
                                     n_subfeatures=[],
                                     k_hashcode=0,
                                     max_k=args.sub_k,
                                     max_radii=[],
                                     n_grids=[],
                                     default_mode=True
                                     )
        predict_func = functools.partial(malscan.predict, n=args.n_sampling, alpha=args.alpha, n_subfeatures=[],
                                         k_per_instance=args.sub_k)
    elif args.model == 'random_malscan':
        input_transfermor = RandomTransformer(keep_per_image=args.sub_k,
                                              reuse_noise=True,  # time-consuming if set reuse_noise to be false
                                              seed=args.seed)
        train_x_train = get_rpst_randomtran(train_x, input_transfermor, args.batch_size, device)
        train_x_producer = torch.from_numpy(train_x_train)
        train_y = torch.from_numpy(train_y).to(device)
        malscan = MalScan(train_x_producer, train_y, args.batch_size)
        malscan = RandomSmooth4MalScan(malscan, number_of_classes=2,
                                       transform_method=input_transfermor,
                                       max_k=args.sub_k,
                                       default_mode=True
                                       )
        predict_func = functools.partial(malscan.predict, n=args.n_sampling, alpha=args.alpha,
                                         k_per_instance=args.sub_k)
    else:
        raise ValueError("Choose either of 'malscan', 'random_smooth', and 'hash_malscan'.\n")

    # test
    _1, _2, test_pkl = dataset.load()
    test_dict, test_y, test_sha256 = test_pkl
    if args.test:
        pred_y = np.empty(shape=(len(test_y, )), dtype=int)
        correct_count = 0
        for i, sha256 in enumerate(test_sha256):
            adj_sp = test_dict[sha256]['adjacent_matrix']
            senstive_node_idx = test_dict[sha256]['sensitive_api_list']
            triple = trans2triple(adj_sp)
            start_time = time.time()
            pred_y[i] = predict_func(x=triple,
                                     adj_size=adj_sp.shape[0],
                                     x_sensitive_dix=senstive_node_idx,
                                     device=device)
            total_time = time.time() - start_time
            print("prediction time: seconds {:.4}.".format(total_time))
            if pred_y[i] == test_y[i]:
                correct_count += 1
            print('correct number: ', correct_count, i + 1)
        mean_acc = np.sum(test_y == pred_y) / len(test_y)
        print("The mean accuracy is {:.4f}%.\n".format(mean_acc * 100))
        logger.info("The mean accuracy is {:.4f}%.\n".format(mean_acc * 100))
    else:
        pass

    # conduct attack for malware examples
    attack_id_path = os.path.join(feature_saving_path, "attack_id.list")
    if not os.path.exists(attack_id_path):
        max_attack_num = 200
        mal_indicator = (test_y == 0)
        mal_test_sha256 = [sha256 for i, sha256 in enumerate(test_sha256) if mal_indicator[i]]
        # remove un-attackable instance
        remove_mal_list = []
        for i in range(len(mal_test_sha256)):
            test_sensti_indix = test_dict[mal_test_sha256[i]]["sensitive_api_list"]
            if np.all(test_sensti_indix == -1):  # no sensitive apis
                remove_mal_list.append(mal_test_sha256[i])
        for sha in remove_mal_list:
            mal_test_sha256.remove(sha)
        np.random.shuffle(mal_test_sha256)
        attack_id = mal_test_sha256[:max_attack_num]
        utils.dump_txt('\n'.join(attack_id), attack_id_path)
    else:
        attack_id = utils.read_txt(attack_id_path)

    print("==== loading test adjacent matrix ====")
    test_adj = [test_dict[sha]["adjacent_matrix"] for sha in tqdm(attack_id)]
    print("==== loading test sensitive api index ====")
    test_sensi_indices = [test_dict[sha]["sensitive_api_list"] for sha in tqdm(attack_id)]
    print("==== loading test constraints ====")
    test_constraints = [test_dict[sha]["constraints"] for sha in tqdm(attack_id)]

    dqn = DQN(states_dim=train_x.shape[1],
              actions_num=ACTION_NUM,
              memory_capacity=args.memory_cap,
              learning_rate=args.lr,
              device=device
              )
    # dqn.load(args.dataset_dir + '/hrat-dpn-model')
    for idx in tqdm(range(0, len(attack_id))):
        test_mal_id = attack_id[idx]
        print("\nAttacking: {}.\n".format(test_mal_id))
        test_mal_adj = test_adj[idx]
        test_sensi_idx = test_sensi_indices[idx]
        triple_path = os.path.join(args.save_path, 'triple_set')
        utils.mkdir(triple_path)
        test_mal_triple = trans2triple_rw(test_mal_adj, test_mal_id, triple_path, overwrite=False)
        test_mal_representation = MalScan.get_extra_feature(test_mal_triple,
                                                            test_sensi_idx,
                                                            adj_size=test_mal_adj.shape[0],
                                                            is_sp2dense=True,
                                                            device=device)
        if test_mal_triple is None:
            logger.info("{}: preprocessing failed.".format(test_mal_id))
            continue

        pred_y = predict_func(x=test_mal_representation,
                              adj_size=test_mal_adj.shape[0],
                              x_sensitive_dix=None,
                              top_k=1,
                              device=device)
        if pred_y == 1:
            print('==== data cannot be correctly classified as malware ====\t')
            logger.info("{}: predict as {}, Attack {}.".format(test_mal_id, pred_y, -1))
            continue

        print('\t ==== select the nearest neighbors for optimization ====')

        X_train_list = torch.split(train_x_producer, args.batch_size)
        dist = torch.cat(
            [torch.sum((test_mal_representation.float() - torch.squeeze(x.to(device))).pow(2), 1) for x in
             X_train_list])

        ben_dist = dist[train_y == 1]
        ben_knn_num = args.knn_num if args.knn_num <= len(ben_dist) else len(ben_dist)
        ben_idx_s = torch.topk(ben_dist, k=ben_knn_num, largest=False)[1].to('cpu')
        ben_train_x = train_x_producer[(train_y == 1).to('cpu')][ben_idx_s]

        mal_dist = dist[train_y == 0]
        mal_knn_num = args.knn_num if args.knn_num <= len(mal_dist) else len(mal_dist)
        mal_idx_s = torch.topk(mal_dist, k=mal_knn_num, largest=False)[1].to('cpu')
        mal_train_x = train_x_producer[(train_y == 0).to('cpu')][mal_idx_s]

        if not args.is_benign:
            train_y_s = torch.zeros((len(ben_train_x),), dtype=int, device=device)
            X_train_sel = ben_train_x
        else:
            train_y_s = torch.zeros((len(ben_train_x) + len(mal_train_x),), dtype=int, device=device)
            train_y_s[:len(ben_train_x)] = 1
            X_train_sel = torch.vstack([ben_train_x, mal_train_x])
        weight = (2 * (train_y_s != 0) - 1).float()
        env = myenv_withconstraints_fs.CFGModifierEnvConstraints(target_graph=test_mal_triple,
                                                                 tar_label=1,
                                                                 target_sen_api_idx=test_sensi_idx,
                                                                 node_num=test_mal_adj.shape[0],
                                                                 w=weight,
                                                                 steep=args.steep,
                                                                 constraints=test_constraints[idx],
                                                                 malware_detector=malscan,
                                                                 predict_function=predict_func,
                                                                 transformer_obj=input_transfermor,
                                                                 n_sampling=args.n_sampling,
                                                                 X_train=X_train_sel,
                                                                 device=device,
                                                                 batch_size=args.batch_size
                                                                 )
        print('\t ==== Attacking... collecting experience ... ====')
        flag = 0
        episode_stop, max_iterations = 5, 100
        best_modifications = 100
        for i_episode in range(15):
            actions_store = []
            state = env.reset()
            ep_r = 0
            count = 0
            while True:
                if dqn.memory_counter <= args.memory_cap:
                    action_type = np.random.randint(ACTION_NUM)
                else:
                    action_type = dqn.choose_action(state, actions_num=ACTION_NUM)
                state_, reward, done, info, cur_graph = env.step(action=action_type)
                if type(action_type) is np.ndarray:
                    action_type = action_type[0]
                action = np.array([action_type] + info)
                dqn.store_transition(state, action, reward, state_, args.memory_cap)
                actions_store.append(action.tolist())
                ep_r += reward
                print("Episode {} w/ round {}: using action {} obtain reward {}/{} w/modification info {}.".format(
                    i_episode,
                    count,
                    action_type,
                    reward,
                    ep_r,
                    info)
                )
                if dqn.memory_counter > args.memory_cap:
                    dqn.learn(args.memory_cap, 16, N_STATES=state.shape[0])
                if done:
                    # check_label = malscan.predict(state_, test_mal_adj.shape[0], device=device)
                    # if check_label == 0:
                    #     logger.warning("something went wrong: check label is {}.".format(check_label))
                    if count < episode_stop:
                        flag = 1
                    if best_modifications >= count:
                        best_modifications = count
                        res_save_path = os.path.join(args.save_path, 'actionseq')
                        utils.mkdir(res_save_path)
                        action_path = res_save_path + "/" + test_mal_id + "action_list" + ".txt"
                        file = open(action_path, 'w')
                        for az in actions_store:
                            file.write(str(az))
                            file.write('\n')
                        file.write(str(done))
                        file.write('\n')
                        file.close()

                        graph_path = res_save_path + "/" + test_mal_id + "graph" + ".npy"
                        np.save(graph_path, cur_graph)

                        feature_file_name = res_save_path + "/" + test_mal_id + "feature_epi" + ".txt"
                        file_feature = open(feature_file_name, 'w')
                        file_feature.write(str(state_.tolist()))
                        file_feature.write('\n')
                        file_feature.write(str(state.tolist()))
                        file_feature.write('\n')
                        file_feature.close()
                        distance_ps = env.getWeightedJaccard(cur_graph, test_mal_triple)
                        distance_fs = env.getWeightedJaccard4Vec(state_.detach().cpu().numpy(),
                                                                 test_mal_representation.detach().cpu().numpy())
                        logger.info(
                            "{}: predict as {}, Attack {} with episode {}, count {}, modification reward {}, jacard distance {} - {}/{}.".format(
                                test_mal_id,
                                pred_y,
                                1,
                                i_episode,
                                count,
                                ep_r,
                                distance_ps,
                                distance_fs,
                                max(test_mal_triple.shape[0], cur_graph.shape[0])))
                    break
                if count >= max_iterations:
                    break
                s = state_
                count += 1
            if flag == 1:
                print('!!!! finish within {}.'.format(episode_stop))
                break
        else:
            logger.info(
                "{}: predict as {}, Attack {}".format(test_mal_id, pred_y, 0))  # Attack failed

        dqn.save(args.dataset_dir + '/hrat-dpn-model')


def get_feature_rpst(file_pkl):
    feature_dict, label, sha256s = file_pkl
    pargs = [(feature_dict[sha256]['adjacent_matrix'], feature_dict[sha256]['sensitive_api_list']) for sha256 in
             sha256s]
    feature_dim = len(feature_dict[sha256s[0]]['sensitive_api_list']) * 2
    feature_x = np.empty(shape=(len(label), feature_dim), dtype=float)
    n_proc = 1 if multiprocessing.cpu_count() // 2 <= 1 else multiprocessing.cpu_count() // 2
    with multiprocessing.Pool(n_proc) as pool:
        for idx, rpst in enumerate(pool.imap(_parallel_featurization, pargs)):
            feature_x[idx] = rpst
    return feature_x, label
    # feature_x = []
    # for sha256 in sha256s:
    #     print("Doing: ", sha256)
    #     adj_sp = feature_dict[sha256]['adjacent_matrix']
    #     node_idx = feature_dict[sha256]['sensitive_api_list']
    #     triple = trans2triple(adj_sp)
    #     feature_x.append(MalScan.get_extra_feature(triple,
    #                                                node_idx,
    #                                                adj_sp.shape[0],
    #                                                device=device).cpu().numpy())
    # return np.array(feature_x), label


def get_rpst_hashtran(train_x, tran_func_obj, batch_size=64, device="cpu"):
    feature_x = []
    num_x = train_x.shape[0]
    current_idx = 0
    for idx in range(num_x // batch_size + 1):
        if current_idx >= num_x - 1:
            break
        batch_train_x = train_x[current_idx: current_idx + batch_size]
        feature_x_batch = np.zeros_like(batch_train_x, dtype=float)
        zero_indicator = np.all(batch_train_x == 0., axis=-1)
        feature_x_batch[~zero_indicator] = \
            tran_func_obj.transform(torch.from_numpy(batch_train_x[~zero_indicator]).to(device),
                                    tran_func_obj.sub_k).cpu().numpy()
        feature_x.append(feature_x_batch)
        current_idx += batch_size
    return np.vstack(feature_x)


def get_rpst_randomtran(train_x, tran_func_obj, batch_size, device):
    feature_x = []
    num_x = train_x.shape[0]
    current_idx = 0
    for idx in range(num_x // batch_size + 1):
        if current_idx >= num_x - 1:
            break
        batch_train_x = train_x[current_idx: current_idx + batch_size]
        feature_x.append(tran_func_obj.transform(torch.from_numpy(batch_train_x).float().to(device),
                                                 tran_func_obj.keep_per_image).cpu().numpy())
        current_idx += batch_size
    return np.vstack(feature_x)


def _parallel_partial_featurization(args):
    triple, adj_sp, node_sens_idx = args
    return MalScan.get_extra_feature(triple, node_sens_idx, adj_sp.shape[0], True, 'cpu').numpy()


def _parallel_featurization(args):
    adj_sp, node_sens_idx = args
    triple = trans2triple(adj_sp)
    return MalScan.get_extra_feature(triple, node_sens_idx, adj_sp.shape[0], True, 'cpu').numpy()


if __name__ == "__main__":
    _main()
