# -*- coding: UTF-8 -*-

# author: Zachary Kaifa ZHAO
# e-mail: kaifa dot zhao (at) connect dot polyu dot hk
# datetime: 2022/3/9 7:33 PM
# software: PyCharm
import os
import re
import warnings
import pickle as pkl
import _pickle as cPickle
import gzip
from scipy import sparse

import numpy as np
import torch

FRAMEWORK = ["java.", "sun.", "android.", "org.apache.", "org.eclipse.", "soot.", "javax.",
             "com.google.", "org.xml.", "junit.", "org.json.", "org.w3c.com"]

LIFECYCLE = ["onCreate", "onClick", "onStart", "onResume", "onPause", "onStop", "onRestart", "onDestroy"]

INIT = ["<clinit>(", "<init>(", "dummyMainClass"]


def degree_centrality_extraction(adjacent_matrix, sen_idx):
    centrality = (adjacent_matrix.sum(axis=0) + adjacent_matrix.sum(axis=1).transpose()) / (
            adjacent_matrix.shape[0] - 1)

    centrality = np.array(centrality)
    centrality = np.squeeze(centrality)
    idx_matrix = np.zeros((len(sen_idx), adjacent_matrix.shape[0]))
    ii = np.where(sen_idx != -1)
    idx_matrix[ii, sen_idx[ii]] = 1
    feature = np.matmul(idx_matrix, centrality)
    return feature


def katz_feature(graph, sen_idx, alpha=0.1, beta=1.0, normalized=True, weight=None):
    graph = graph.T
    n = graph.shape[0]
    b = np.ones((n, 1)) * float(beta)
    centrality = np.linalg.solve(np.eye(n, n) - (alpha * graph), b)
    if normalized:
        norm = np.sign(sum(centrality)) * np.linalg.norm(centrality)
    else:
        norm = 1.0
    centrality = centrality / norm
    idx_matrix = np.zeros((len(sen_idx), n))
    ii = np.where(sen_idx != -1)
    idx_matrix[ii, sen_idx[ii]] = 1
    feature = np.matmul(idx_matrix, centrality)
    return feature


def trans2triple_rw(adjacent_matrix, sha256, triple_path, overwrite=False, max_v=10.):
    file_name = triple_path + "/" + sha256 + ".npy"
    triple = []

    if os.path.exists(file_name) and not overwrite:
        print("loading")
        triple = np.load(file_name)
        if triple.shape[0] < adjacent_matrix.shape[0]:
            triple = trans2triple_rw(
                adjacent_matrix, sha256, triple_path, overwrite=True, max_v=max_v)
    else:
        node_num = adjacent_matrix.shape[1]
        # triple = []
        # node_number = adjacent_matrix.shape[0]
        #
        # if type(adjacent_matrix) is sparse.coo_matrix:
        #     adjacent_matrix = adjacent_matrix.tocsr()
        # for zi in range(node_number):
        #     # triple.append([zi, zi, adjacent_matrix[zi, zi]])
        #     for zj in range(zi + 1, node_number):
        #         triple.append([zi, zj, adjacent_matrix[zi, zj]])
        #         triple.append([zj, zi, adjacent_matrix[zj, zi]])
        #
        # triple = np.array(triple)

        adjacent_matrix = sparse.coo_matrix(adjacent_matrix.toarray())
        r, c = adjacent_matrix.row, adjacent_matrix.col
        pos = np.vstack([r, c]).T
        pos_one_element = np.concatenate([pos, np.array(adjacent_matrix.data)[..., None]], axis=-1).clip(0., max_v)
        r, c = np.where(adjacent_matrix.toarray() == 0)
        pos = np.vstack([r, c]).T
        pos_zero_element = np.concatenate([pos, np.zeros((len(pos), 1), dtype=int)], axis=-1)
        triple = np.vstack([pos_one_element, pos_zero_element])

        diag_indicator = triple[:, 0] == triple[:, 1]
        triple = triple[~diag_indicator]

        # node_number = adjacent_matrix.shape[0]
        # triple = []
        # if type(adjacent_matrix) is sparse.coo_matrix:
        #     adjacent_matrix = adjacent_matrix.tocsr()
        #
        # count_num = len(r)
        # for zi in range(node_number):
        #     # triple.append([zi, zi, adjacent_matrix[zi, zi]])
        #     for zj in range(zi + 1, node_number):
        #         if np.any(np.all(np.array([zi, zj]) == pos, axis=1)):
        #             idc = np.all(np.array([zi, zj]) == pos, axis=1)
        #             print("ok", pos[idc])
        #             continue
        #         count_num += 1
        #         print(count_num)
        #         if count_num >= 100000:  # D. Li add this line
        #             break
        #         triple.append([zi, zj, adjacent_matrix[zi, zj]])
        #         triple.append([zj, zi, adjacent_matrix[zj, zi]])
        # triple = np.array(triple)
        # triple = np.vstack([pos_one_element, triple])
        np.save(file_name, triple)
    return triple


def trans2triple(adjacent_matrix, max_v=10.):
    # triple = []
    # node_number = adjacent_matrix.shape[0]
    # if type(adjacent_matrix) is sparse.coo_matrix:
    #     adjacent_matrix = adjacent_matrix.tocsr()
    # for zi in range(node_number):
    #     # triple.append([zi, zi, adjacent_matrix[zi, zi]])
    #     for zj in range(zi + 1, node_number):
    #         triple.append([zi, zj, adjacent_matrix[zi, zj]])
    #         triple.append([zj, zi, adjacent_matrix[zj, zi]])
    # triple = np.array(triple)
    adjacent_matrix = sparse.coo_matrix(adjacent_matrix.toarray())
    r, c = adjacent_matrix.row, adjacent_matrix.col
    pos = np.vstack([r, c]).T
    pos_one_element = np.concatenate([pos, np.array(adjacent_matrix.data)[..., None]], axis=-1).clip(0, max_v)
    r, c = np.where(adjacent_matrix.toarray() == 0)
    pos = np.vstack([r, c]).T
    pos_zero_element = np.concatenate([pos, np.zeros((len(pos), 1), dtype=int)], axis=-1)
    triple = np.vstack([pos_one_element, pos_zero_element])

    diag_indicator = triple[:, 0] == triple[:, 1]
    triple = triple[~diag_indicator]
    return triple


def find_nn_torch(Q, X, y, k=1, spliter=64):
    if isinstance(X, torch.Tensor):
        X = torch.split(X, spliter)
    dist = torch.cat([torch.sum((torch.squeeze(x).to(Q.device) - torch.squeeze(Q)).pow(2.), 1) for x in X])
    # dist = torch.sum((torch.squeeze(X) - torch.squeeze(Q)).pow(2.), 1)
    ind = torch.argsort(dist)
    label = y[ind[:k]]

    label_total = y[ind]
    list_label = label_total.cpu().numpy()

    benign_idx = np.argwhere(list_label == 1)
    min_dist = dist[ind[benign_idx[0][0]]]

    unique_label = torch.unique(y)
    unique_label = unique_label.long()
    count = np.zeros(unique_label.shape[0])
    for i in label:
        count[unique_label[i.long()]] += 1
    ii = torch.argmax(torch.from_numpy(count))
    final_label = unique_label[ii]
    return final_label, min_dist


def to_adjmatrix(adj_sparse, adj_size):
    A = torch.sparse_coo_tensor(adj_sparse[:, :2].T, adj_sparse[:, 2],
                                size=[adj_size, adj_size]).to_dense()
    return A


def degree_centrality_torch(adj, sen_api_idx, device='cuda:0'):
    adj_size = adj.shape[0]
    idx_matrix = np.zeros((len(sen_api_idx), adj_size))
    ii = np.where(sen_api_idx != -1)
    idx_matrix[ii, sen_api_idx[ii]] = 1
    idx_matrix = torch.from_numpy(idx_matrix).to(device)

    if adj.shape[0] > len(idx_matrix):
        _sub = adj.shape[0] - len(idx_matrix)
        for za in range(_sub):
            idx_matrix.append[0]

    all_degree = torch.div((torch.sum(adj, 0) + torch.sum(adj, 1)).float(),
                           float(adj.shape[0] - 1))
    degree_centrality = torch.matmul(
        idx_matrix, all_degree.type_as(idx_matrix))
    return degree_centrality


def katz_feature_torch(graph, sen_api_idx, alpha=0.1, beta=1.0, device='cuda:0', normalized=True):
    n = graph.shape[0]
    graph = graph.T
    b = torch.ones((n, 1)) * float(beta)
    b = b.to(device)
    graph = graph.to(device)
    A = torch.eye(n, n).to(device).float() - (alpha * graph.float())
    L, U = torch.solve(b, A)
    if normalized:
        norm = torch.sign(sum(L)) * torch.norm(L)
    else:
        norm = 1.0
    centrality = torch.div(L, norm.to(device)).to(device)
    idx_matrix = np.zeros((len(sen_api_idx), n))
    ii = np.where(sen_api_idx != -1)
    idx_matrix[ii, sen_api_idx[ii]] = 1
    idx_matrix = torch.from_numpy(idx_matrix).to(device)
    katz_centrality = torch.matmul(idx_matrix, centrality.type_as(idx_matrix))
    return katz_centrality


def obtain_sensitive_apis(file):
    sensitive_apis = []
    with open(file, 'r') as f:
        for line in f.readlines():
            if line.strip() == '':
                continue
            else:
                sensitive_apis.append(line.strip())
    return sensitive_apis


smaliBasicT_javaBasicT = {
    'boolean': 'Z',
    'byte': 'B',
    'short': 'S',
    'char': 'C',
    'int': 'I',
    'float': 'F',
    'long': 'J',
    'double': 'D',
    'void': 'V'
}
javaBasicT_smaliBasicT = dict([(v, k) for k, v in smaliBasicT_javaBasicT.items()])


def get_java_classname(smali_classname):
    if '/' in smali_classname:
        class_name = smali_classname.replace('/', '.')
    if class_name.startswith('L'):
        class_name = class_name.lstrip('L')
    if class_name.endswith(';'):
        class_name = class_name.rstrip(';')
    return class_name


def parse_element(param):
    array_flag = False
    if param.startswith('['):
        array_flag = True
        param = param.lstrip('[')
    if param.startswith('L') and param.endswith(';'):  # class
        java_element = get_java_classname(param)
    elif param in smaliBasicT_javaBasicT.values():
        java_element = javaBasicT_smaliBasicT[param]
    else:
        java_element = ''
        warnings.warn("Cannot parse {}.\n".format(param))
    if array_flag:
        java_element += '[]'
    return java_element


def get_param_smali_type(params):
    """Get arugments' smali type of specified params which is used by a method"""
    param_types_smali = []
    if params == '':
        return param_types_smali

    class_param_types = params.split(';')
    for pt in class_param_types:  # type: string
        pt = pt.strip().replace(' ', '')
        if pt == '':
            continue

        tmp_prefix = []
        for i, c in enumerate(pt):
            if c in smaliBasicT_javaBasicT.values():
                if len(tmp_prefix) > 0:
                    param_types_smali.append(''.join(tmp_prefix) + c)
                    tmp_prefix = []
                else:
                    param_types_smali.append(c)
            elif c == '[':
                tmp_prefix.append(c)
            elif c == 'L':
                if len(tmp_prefix) > 0:
                    param_types_smali.append(''.join(tmp_prefix) + pt[i:] + ";")
                else:
                    param_types_smali.append(pt[i:] + ";")
                break
            else:
                raise ValueError("The symbol '{}' are negected".format(str(c)))
    return param_types_smali


def get_tran_sensitive_apis(file):
    sensitive_apis = obtain_sensitive_apis(file)
    sensitive_apis_ = []
    for api in sensitive_apis:
        api_match = re.search(
            r"(?P<invokeObject>L(.*?);|\[L(.*?);)->(?P<invokeMethod>(.*?))\((?P<invokeArgument>(.*?))\)(?P<invokeReturn>(.*?))$",
            api
        )
        if api_match is None:
            warnings.warn("Miss match the api {}".format(api))
        else:
            smali_class_name = api_match['invokeObject']
            java_class_name = get_java_classname(smali_class_name)

            smali_method_name = api_match['invokeMethod']

            smali_argument = api_match['invokeArgument']
            arg_elements = get_param_smali_type(smali_argument)
            java_arguments = ','.join([parse_element(arg_e) for arg_e in arg_elements])

            smali_return = api_match['invokeReturn']
            java_return = parse_element(smali_return)

            new_api = '<' + java_class_name + ': ' + java_return + ' ' + smali_method_name + '(' + java_arguments + ')' + '>'
            sensitive_apis_.append(new_api)
    if len(sensitive_apis_) == len(sensitive_apis):
        print("Done!\n")
    else:
        warnings.warn("Some apis are absent.")
    return sensitive_apis_


def extract_sensitive_api(sensitive_api_list, nodes_list):
    sample_sensitive_api = []
    for x in sensitive_api_list:
        if x in nodes_list:
            sample_sensitive_api.append(nodes_list.index(x))
        else:
            sample_sensitive_api.append(-1)
    return np.array(sample_sensitive_api)


def check_folder(folder_name):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)


def check_folder(folder_name: str):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)


def constrains_amend(node_list, constraints):
    new_constraints = []
    for node, cons in zip(node_list, constraints):
        if cons == -1:
            new_constraints.append(-1)
        elif cons == 0:
            flag = False
            for lifecycle_api in LIFECYCLE:
                if lifecycle_api in node:
                    new_constraints.append(0)
                    flag = True
                    break
            for init_api in INIT:
                if init_api in node:
                    new_constraints.append(0)
                    flag = True
                    break
            if not flag:
                new_constraints.append(-1)
        else:
            new_constraints.append(-2)
    return new_constraints


def fcg_to_adjacent(node_file, fcg_file, constraint_file, sensitive_apis=None, limited_node_num=10000):
    tmp_node = open(node_file, "r", encoding='utf-8')
    node_list = [zn.replace("\n", "") for zn in tmp_node.readlines()]
    tmp_node.close()

    tmp_cons = open(constraint_file, 'r', encoding='utf-8')
    constraints = [int(a.replace("\n", "")) for a in tmp_cons.readlines()]
    tmp_cons.close()

    constraints = constrains_amend(node_list, constraints)

    if 0 < limited_node_num < len(node_list):
        sensitive_nodes, non_senst_nodes = [], []
        sensitive_cons, non_sensitive_cons = [], []
        for i, node in enumerate(node_list):
            if node in sensitive_apis:
                sensitive_nodes.append(node)
                sensitive_cons.append(constraints[i])
            else:
                non_senst_nodes.append(node)
                non_sensitive_cons.append(constraints[i])
        if len(sensitive_nodes) - limited_node_num > 0:
            node_list = sensitive_nodes
            constraints = sensitive_cons
        else:
            node_list = sensitive_nodes + non_senst_nodes[:limited_node_num - len(sensitive_nodes)]
            constraints = sensitive_cons + non_sensitive_cons[:limited_node_num - len(sensitive_nodes)]
    else:
        pass

    tmp_graph = open(fcg_file, "r", encoding='utf-8')
    line = tmp_graph.readline()
    row_ind = []
    col_ind = []
    data = []
    while line:
        # print(line)
        line = line.split("\n")[0]
        nodes = line.split(" ==> ")
        if nodes[0] not in node_list or nodes[1] not in node_list:
            break
        row_ind.append(node_list.index(nodes[0]))
        col_ind.append(node_list.index(nodes[1]))
        data.append(1)
        line = tmp_graph.readline()
    tmp_graph.close()
    adj_matrix = sparse.coo_matrix((data, (row_ind, col_ind)), shape=[len(node_list), len(node_list)])
    return adj_matrix, node_list, constraints


def load_constraints(cons_file):
    f = open(cons_file, "r", encoding='utf-8').readlines()
    constraints = [int(a.replace("\n", "")) for a in f]
    constraints = np.array(constraints)
    return constraints


def adj_to_triple(adj):
    """
    :param adj: adjacent matrix
    :return: triple set -- numpy array -- [row_index, col_index, edge_state]
    """
    if type(adj) is sparse.coo_matrix:
        adjacent_matrix = adj.tocsr()

    node_number = adjacent_matrix.shape[0]
    triple = []
    for zi in range(node_number):
        # triple.append([zi, zi, adjacent_matrix[zi, zi]])
        for zj in range(zi + 1, node_number):
            triple.append([zi, zj, adjacent_matrix[zi, zj]])
            triple.append([zj, zi, adjacent_matrix[zj, zi]])
    return np.array(triple)


def get_subset_of_training_set(test_feature, X_train, m):
    test_feature_tmp = np.array(test_feature)[np.newaxis, :]
    axis = tuple(np.arange(1, X_train.ndim, dtype=np.int32))
    dist = np.sum(np.square(X_train - test_feature_tmp), axis=axis)
    dist = np.squeeze(dist)
    ind = np.argsort(dist)
    ind = np.squeeze(ind).transpose()
    return ind[:m]


def mkdir(target):
    try:
        if os.path.isfile(target):
            target = os.path.dirname(target)

        if not os.path.exists(target):
            os.makedirs(target)
        return 0
    except IOError as e:
        raise Exception("Fail to create directory! Error:" + str(e))


def dump_pickle(data, path, use_gzip=False):
    if not os.path.exists(os.path.dirname(path)):
        mkdir(os.path.dirname(path))
    if not use_gzip:
        with open(path, 'wb') as wr:
            pkl.dump(data, wr)
    else:
        with gzip.open(path, 'wb') as wr:
            pkl.dump(data, wr)
    return True


def read_pickle(path, use_gzip=False):
    if os.path.isfile(path):
        if not use_gzip:
            with open(path, 'rb') as fr:
                return pkl.load(fr)
        else:
            with gzip.open(path, 'rb') as fr:
                return pkl.load(fr)
    else:
        raise IOError("The {0} is not been found.".format(path))


class MalscanDataset(torch.utils.data.Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        x = self.data[item]
        y = self.label[item]
        return x, y

