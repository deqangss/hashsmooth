# -*- coding: UTF-8 -*-

# author: Deqiang Li
# datetime: 2023/8/7 7:42 PM
# software: PyCharm
import time
import os
import warnings
import argparse
import logging
from tqdm import tqdm
import multiprocessing

import numpy as np
import torch

from sec_classifiers.hrat_malscan.hrat_malscan_det import MalScan
from tools import utils
from sec_classifiers.hrat_malscan.Utils import trans2triple_rw, trans2triple
from sec_classifiers.hrat_malscan import myenv_withconstraints_dli
from sec_classifiers.hrat_malscan.model import DQN
from sec_classifiers.dataset import Dataset

ACTION_NUM = 4

torch.manual_seed(23456)
torch.cuda.manual_seed(23456)
np.random.seed(23456)

cmd_md = argparse.ArgumentParser(description='arguments for hrat attack')

cmd_md.add_argument('--memory_cap', type=int, default=16,
                    help='The memory capability of DQN.')
cmd_md.add_argument('--lr', type=float, default=0.01,
                    help='The learning rate of DQN.')
cmd_md.add_argument('--steep', type=float, default=1.,
                    help='The hyper-parameter of surrogate KNN.')
cmd_md.add_argument('--max_node', type=int, default=10000,
                    help='The maximum number of nodes for saving RAM')
cmd_md.add_argument('--seed', type=int, default=23456,
                    help='Random seed for reproduction')
cmd_md.add_argument('--cuda', action='store_true', default=False,
                    help='whether use cuda enable gpu or cpu.')
cmd_md.add_argument('--dataset_dir', type=str, default='',
                    help='Folder path to dataset directory.')
cmd_md.add_argument('--dataset_name', type=str, default='malscan',
                    help='Dataset name.')
cmd_md.add_argument('--batch_size', type=int, default=64,
                    help='computation upon a batch of data instances for saving RAM')
cmd_md.add_argument('--test', action='store_true', default=False,
                    help='Predict labels for all test data instances.')
cmd_md.add_argument('--save_path', type=str, default='./results',
                    help='Folder path to save results.')


def _main():
    args = cmd_md.parse_args()
    if not os.path.exists(args.save_path):
        utils.mkdir(args.save_path)
    logging.basicConfig(level=logging.INFO,
                        filename=os.path.join(args.save_path, time.strftime("%Y%m%d-%H%M%S") + ".log"),
                        filemode="w",
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s',
                        datefmt='%Y/%m/%d %H:%M:%S')
    ErrorHandler = logging.StreamHandler()
    ErrorHandler.setFormatter(logging.Formatter('%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s'))

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

    train_x_producer = dataset.get_dataloader(train_x)
    malscan = MalScan(train_x_producer, torch.from_numpy(train_y).to(device))

    # test
    _1, _2, test_pkl = dataset.load()
    test_dict, test_y, test_sha256 = test_pkl
    if args.test:
        pred_y = np.empty(shape=(len(test_y, )))
        for i, sha256 in enumerate(test_sha256):
            print("predict: ", sha256)
            adj_sp = test_dict[sha256]['adjacent_matrix']
            senstive_node_idx = test_dict[sha256]['sensitive_api_list']
            triple = trans2triple(adj_sp)
            start_time = time.time()
            pred_y[i] = malscan.predict(triple, adj_sp.shape[0], x_sensitive_dix=senstive_node_idx, device='cpu')
            total_time = time.time() - start_time
            print("prediction time: secondes {:.4}.".format(total_time))

        mean_acc = np.sum(test_y == pred_y) / len(test_y)
        print("The mean accuracy is {:.4f}%.\n".format(mean_acc * 100))
        logging.info("The mean accuracy is {:.4f}%.\n".format(mean_acc * 100))
    else:
        pass

    # conduct attack for malware examples
    attack_id_path = os.path.join(feature_saving_path, "attack_id.list")
    if not os.path.exists(attack_id_path):
        max_attack_num = 200
        mal_indicator = (test_y == 0)
        mal_test_sha256 = [sha256 for i, sha256 in enumerate(test_sha256) if mal_indicator[i]]
        # remove unattackable instance
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
    for idx in tqdm(range(0, len(attack_id))):
        test_mal_id = attack_id[idx]
        print("Attacking: ", test_mal_id)
        test_mal_adj = test_adj[idx]
        test_sensi_idx = test_sensi_indices[idx]
        triple_path = os.path.join(args.save_path, 'triple_set')
        utils.mkdir(triple_path)
        test_mal_triple = trans2triple_rw(test_mal_adj, test_mal_id, triple_path, overwrite=False)
        if test_mal_triple is None:
            logging.info("{}: preprocessing failed.".format(test_mal_id))
            continue

        pred_y = malscan.predict(test_mal_triple, test_mal_adj.shape[0],
                                 x_sensitive_dix=test_sensi_idx, device='cpu')
        if pred_y != 0:
            print('==== data cannot be correctly classified as malware ====\t')
            logging.info("{}: predict as {}, Attack {}.".format(test_mal_id, pred_y, -1))
            continue

        print('\t ==== get the nearest neighbors for optimization ====')
        weight = (2 * (torch.from_numpy(train_y) != 0).int() - 1).float().to(device)
        env = myenv_withconstraints_dli.CFGModifierEnvConstraints(target_graph=test_mal_triple,
                                                                  label=0,
                                                                  target_sen_api_idx=test_sensi_idx,
                                                                  node_num=test_mal_adj.shape[0],
                                                                  w=weight,
                                                                  steep=args.steep,
                                                                  constraints=test_constraints[idx],
                                                                  malware_detector=malscan,
                                                                  device=device
                                                                  )
        print('\t ==== Attacking...collecting experience ... ====')
        flag = 0
        episode_stop, max_iterations = 5, 100
        best_modifications = 100
        for i_episode in range(10):
            actions_store = []
            state = env.reset()
            ep_r = 0
            count = 0
            while True:
                if dqn.memory_counter <= args.memory_cap:
                    action_type = np.random.randint(ACTION_NUM)
                else:
                    action_type = dqn.choose_action(state, actions_num=ACTION_NUM)
                start_time = time.time()
                state_, reward, done, info, cur_graph = env.step(action=action_type,
                                                                 X_train=train_x_producer,
                                                                 y_train=torch.from_numpy(train_y))
                total_time = time.time() - start_time
                print("prediction time: secondes {:.4}.".format(total_time))
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
                    check_label = malscan.predict(state_, test_mal_adj.shape[0], device=device)
                    if check_label == 0:
                        logging.warning("something went wrong: check label is {}.".format(check_label))
                        print("okok")
                        exit(-1)
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
                        distance = env.getWeightedJaccard(cur_graph, test_mal_triple)
                        logging.info(
                            "{}: predict as {}, Attack {} with episode {}, count {}, modification reward {}, jacard distance {}/{}.".format(
                                test_mal_id,
                                pred_y,
                                1,
                                i_episode,
                                count,
                                ep_r,
                                distance,
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
            logging.info(
                "{}: predict as {}, Attack {}".format(test_mal_id, pred_y, 0))  # Attack failed


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


def _parallel_featurization(args):
    adj_sp, node_sens_idx = args
    triple = trans2triple(adj_sp)
    return MalScan.get_extra_feature(triple, node_sens_idx, adj_sp.shape[0], True, 'cpu').cpu().numpy()


if __name__ == "__main__":
    _main()