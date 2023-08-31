import time
import numpy as np
import torch
from torch.autograd import Variable
import pandas as pd

from sec_classifiers.hrat_malscan.hrat_malscan_det import MalScan
from sec_classifiers.hrat_malscan.hashsmooth_hrat_malscan_det import HashSmooth4MalScan


class CFGModifierEnvConstraints(object):
    """
    Description:

    Source:
        Self designed. This environments is designed to modify function call graph in programming.

    Observations:
        Type: vector:
            the centrality of sensitive apis (attention nodes)

    Actions:
        Type: list(4)
        Num     Action
        0       add an edge between two nodes
        1       rewiring
        2       add an nodes, and connect it to another nodes
        3       delete an nodes


    Reward:
        Reward is


    Episode Termination:
        The observation is classified as target class.
        Episode length is greater than 200.

    """

    # torch.manual_seed(6666)
    # np.random.seed(6666)

    def __init__(self, target_graph, label,
                 target_sen_api_idx, node_num,
                 w, steep, constraints,
                 malware_detector,
                 predict_function,
                 X_train,
                 batch_size=64,
                 device='cpu'):
        self.adj_sparse = target_graph
        self.adj_size_ori = node_num
        self.label = label
        self.adj_size = node_num
        self.sen_api_idx = target_sen_api_idx
        self.sen_api_idx_ori = target_sen_api_idx
        self.w = w
        self.steep = steep
        self.constraints = constraints
        self.constraints_ori = constraints
        self.action_space = ['add_edge', 'rewiring', 'add_nodes', 'delete_nodes']
        self.cur_graph = target_graph
        self.malware_detector = malware_detector
        self.predict_function = predict_function
        self.device = device
        self.batch_size = batch_size
        if isinstance(X_train, torch.Tensor):
            self.X_train = torch.split(torch.squeeze(X_train), self.batch_size)
        elif isinstance(X_train, torch.utils.data.dataloader.DataLoader):
            self.X_train = X_train
        else:
            raise ValueError

    def step(self, action, k=1):
        assert len(self.action_space) >= action + 1
        print("Current action: ", action)
        if action == 0:  # add dege
            self.cur_graph, node_info = self.add_edge2(self.cur_graph)

        elif action == 1:  # rewiring
            self.cur_graph, node_info = self.rewiring(self.cur_graph)

        elif action == 2:  # add node
            self.cur_graph, node_info = self.add_node(self.cur_graph)

        elif action == 3:  # delete node
            self.cur_graph, node_info = self.del_node(self.cur_graph)
        else:
            raise ValueError("No action {}.\n".format(action))

        check_flag = self.cur_graph[:, 0] == self.cur_graph[:, 1]
        if np.any(check_flag):
            print(self.cur_graph[check_flag])
            exit(-1)

        with torch.no_grad():
            self.state = MalScan.get_extra_feature(self.cur_graph,
                                                   self.sen_api_idx,
                                                   adj_size=self.adj_size,
                                                   is_sp2dense=True,
                                                   device=self.device)
            if isinstance(self.malware_detector, HashSmooth4MalScan):
                cur_label = self.predict_function(x=self.cur_graph,
                                                  adj_size=self.adj_size,
                                                  top_k=k,
                                                  x_sensitive_dix=self.sen_api_idx,
                                                  device=self.device)
            elif isinstance(self.malware_detector, MalScan):
                cur_label = self.predict_function(x=self.state,
                                                  adj_size=self.adj_size,
                                                  top_k=k,
                                                  device=self.device)
            else:
                raise TypeError
        done = (cur_label != self.label)

        if done:
            reward = 10
        else:
            # todo ADD THE DISTANCE OF CUR_GRAPH TO NEWAREST BENIGH
            reward = self.getReward(self.cur_graph)
            if action == 1:
                if reward == 0:
                    reward = -2.0
                else:
                    reward = -3.0
            else:
                if reward == 0 and node_info == [-1, -1, -1]:
                    reward = -3.0
        return self.state, reward, done, node_info, self.cur_graph

    def reset(self):
        with torch.no_grad():
            self.adj_size = self.adj_size_ori
            self.cur_graph = self.adj_sparse.copy()
            self.sen_api_idx = self.sen_api_idx_ori.copy()
            self.constraints = self.constraints_ori.copy()
            self.state = MalScan.get_extra_feature(self.adj_sparse, self.sen_api_idx,
                                                   self.adj_size,
                                                   True,
                                                   device=self.device)
            self.last_n_edge = np.where(self.cur_graph[:, -1] != 0)[0].shape[0]
            self.last_n_node = max(np.unique(self.cur_graph[:, 0]).shape[0], np.unique(self.cur_graph[:, 1]).shape[0])
        return self.state

    def getReward(self, graph, budget_node=1, budget_edge=1):
        n_edge = np.where(graph[:, -1] != 0)[0].shape[0]
        n_node = max(np.unique(graph[:, 0]).shape[0], np.unique(graph[:, 1]).shape[0])

        if n_edge == self.last_n_edge:
            edge_r = 0
        else:
            edge_r = 1 / budget_edge * (self.last_n_edge - n_edge)

        if n_node == self.last_n_node:
            node_r = 0
        else:
            node_r = 1 / budget_node * (self.last_n_node - n_node)

        reward = -(abs(edge_r) + abs(node_r))
        return reward

    def getWeightedJaccard(self, graph, last_graph, budget_node=1, budget_edge=1):
        graph_zero_indicator = graph[:, -1] == 0
        graph_filled = graph[~graph_zero_indicator]
        graph_pos = tuple(map(tuple, graph_filled[:, :2]))
        graph_dict = dict(zip(graph_pos, graph_filled[:, -1].tolist()))

        last_graph_zero_indicator = last_graph[:, -1] == 0
        last_graph_filled = last_graph[~last_graph_zero_indicator]
        last_graph_pos = tuple(map(tuple, last_graph_filled[:, :2]))
        last_graph_dict = dict(zip(last_graph_pos, last_graph_filled[:, -1].tolist()))

        all_positions = list(set(list(graph_dict.keys()) + list(last_graph_dict.keys())))
        sum_min, sum_max = 0., 0.
        for pos in all_positions:
            if pos not in graph_dict.keys():
                graph_v = 0.
            else:
                graph_v = graph_dict[pos]
            if pos not in last_graph_dict.keys():
                last_graph_v = 0.
            else:
                last_graph_v = last_graph_dict[pos]
            assert graph_v != 0. or last_graph_v != 0.
            sum_min += min(graph_v, last_graph_v)
            sum_max += max(graph_v, last_graph_v)
        sum_max = 1e-6 if sum_max == 0 else sum_max
        return 1 - sum_min / sum_max

    def del_node(self, graph):
        # cal grad of all edges
        tmp_grad = self.get_gradient2(graph, is_dense=True)
        node_grad = (torch.sum(tmp_grad, 0) + torch.sum(tmp_grad, 1)).cpu().numpy()
        node_id = np.arange(node_grad.shape[0])
        # sort node_grad
        # grad_tmp = np.array([node_id.tolist(), node_grad.tolist()]).transpose()
        grad_tmp = np.stack([node_id, node_grad]).T
        a = grad_tmp[grad_tmp[:, -1].argsort()]
        a = a[::-1]
        tar_node = -1
        for zi in a:
            flag = 0
            if self.constraints[int(zi[0])] == 0 or self.constraints[int(zi[0])] == -1 or zi[
                -1] < 0:  # caller cannot be contained in constraints; -1为新constraints
                flag = 1
                if zi[-1] < 0:
                    break
            # find functions that call target nodes
            ind_caller = np.where(graph[:, 0] == int(zi[0]))  # 应该是graph[:,1]吧
            caller_idx = graph[ind_caller, 0]
            # true_ind_caller = np.where(caller_idx[:, 2] == 1)
            # true_caller_idx = np.squeeze(caller_idx[true_ind_caller,0], axis=0)
            for zi2 in caller_idx:
                if self.constraints[int(zi2[0])] == 0:  # caller cannot be contained in constraints
                    flag = 1
                    break
            if flag == 1:
                continue
            else:
                tar_node = int(zi[0])
                break
        if tar_node < 0:
            print("zkf no nodes to delete")
            return graph, [-1, -1, -1]
        # find edges to tar node
        tmp_ind_to = np.where(graph[:, 1] == tar_node)[0]
        edge_to_tar_node = graph[tmp_ind_to, :]
        # edge_to_tar_node = np.squeeze(edge_to_tar_node)
        tmp_ind = np.where(edge_to_tar_node[:, 2] == 1)[0]
        edge_to_tar_node = edge_to_tar_node[tmp_ind, :]

        # find edges from tar node
        tmp_ind_from = np.where(graph[:, 0] == tar_node)[0]
        edge_from_tar_node = graph[tmp_ind_from, :]
        edge_from_tar_node = np.squeeze(edge_from_tar_node)
        tmp_ind = np.where(edge_from_tar_node[:, 2] == 1)
        edge_from_tar_node = np.squeeze(edge_from_tar_node[tmp_ind, :], axis=0)

        # print(edge_to_tar_node, edge_from_tar_node)

        # 如果该节点调用了其他节点，且该节点存在调用
        if edge_from_tar_node.size != 0 and edge_to_tar_node.size != 0:
            # 对于所有调用了改节点的函数
            for zind_beg in edge_to_tar_node[:, 0]:
                tmp_ind = np.where(graph[:, 0] == zind_beg)
                tmp_ind = np.squeeze(tmp_ind)
                edge_tmp = np.squeeze(graph[tmp_ind, :])
                # 对于所有该结点调用的结点
                for zind_end in edge_from_tar_node[:, 1]:
                    tmp_ind1 = np.where(edge_tmp[:, 1] == zind_end)
                    tmp_ind1 = np.squeeze(tmp_ind1)
                    if tmp_ind1.size != 0:
                        # edge = tmp_ind[tmp_ind1]
                        # ii = np.where(np.all((graph[:, :2] == edge), axis=1) == True)
                        # graph[ii, 2] = 1
                        graph[tmp_ind[tmp_ind1], 2] = 1
                    else:
                        # graph.append([zind_beg, zind_end, 1])
                        graph = np.append(graph, np.array([zind_beg, zind_end, 1])[:, np.newaxis].transpose(), axis=0)
        # del nodes
        tmp_ind_to = np.where(graph[:, 1] == tar_node)
        graph = np.delete(graph, tmp_ind_to, axis=0)
        tmp_ind_from = np.where(graph[:, 0] == tar_node)
        graph = np.delete(graph, tmp_ind_from, axis=0)

        # graph[np.where(graph[:, 1] > tar_node), 1] -= 1
        # graph[np.where(graph[:, 0] > tar_node), 0] -= 1

        graph[graph[:, 1] > tar_node, 1] -= 1
        graph[graph[:, 0] > tar_node, 0] -= 1

        # self.sen_api_idx[np.where(self.sen_api_idx > tar_node)[0]] -= 1
        self.sen_api_idx[self.sen_api_idx > tar_node] -= 1

        self.constraints = np.delete(self.constraints, tar_node)
        self.adj_size -= 1

        # print("len con:", str(len(self.constraints)), "  size graph:", str(max(graph[:, 0])), ',',
        #       str(max(graph[:, 1])),
        #       ', sen_api_idx:', str(np.max(self.sen_api_idx)))
        # print("\t\t graph idx:", np.min(graph[:, 0]), ", ", np.min(graph[:, 1]), ", ", np.min(graph[:, 2]), ", ")

        return graph, [-3, -1, tar_node]

    def add_node(self, graph):
        self.adj_size += 1
        # add one nodes to the sparse adjacent matrix
        a = np.arange(self.adj_size - 1)  # [i for i in range(self.adj_size - 1)]
        ii = np.ones_like(a)
        # 在adj中添加新的节点
        tmp_row = [ii * (self.adj_size - 1), a, (ii - 1)]
        tmp_row = np.array(tmp_row).transpose()
        tmp_col = [a, ii * (self.adj_size - 1), (ii - 1)]
        tmp_col = np.array(tmp_col).transpose()
        tmpz = np.concatenate((graph, tmp_col, tmp_row), axis=0)
        grad = self.get_gradient2(tmpz)

        # find the edge with max graient and add the edge
        # idx = np.where(grad[:, 1] == self.adj_size - 1)[0]
        grad_tmp = grad[grad[:, 1] == self.adj_size - 1, :]
        # grad_tmp = np.squeeze(grad_tmp)

        # sort grad_tmp
        a = grad_tmp[grad_tmp[:, 2].argsort()]
        # a = a[::-1]
        edge = []
        for zi in a:
            # if int(zi[0]) > len(self.constraints) - 1:
            #     continue
            if self.constraints[int(zi[0])] == 0:  # caller cannot be contained in constraints
                continue
            elif zi[-1] > 0:
                break
            else:
                edge = zi[:2].astype(np.int64)
                break
        if edge == []:
            self.adj_size -= 1
            return graph, [-1, -1, -1]  # cannot find a node for adding

        ii = np.where(np.all((tmpz[:, :2] == edge), axis=1) == True)[0]
        tmpz[ii, 2] = 1
        # self.cur_graph = tmpz
        self.constraints = np.append(self.constraints, 1)
        # print("len con:", str(len(self.constraints)),
        #       "  size graph:", str(max(tmpz[:, 0])), ',', str(max(tmpz[:, 1])),
        #       ', sen_api_idx:', str(np.max(self.sen_api_idx)))

        return tmpz, [int(np.squeeze(tmpz[ii, 0])), int(np.squeeze(tmpz[ii, 1])), self.adj_size - 1]

    def add_edge(self, graph):
        # triple_copy = graph.copy()
        # tmpz = torch.from_numpy(triple_copy).float()
        # triple_torch = Variable(tmpz, requires_grad=True)
        grad = self.get_gradient2(graph)
        # get the add edge with max grad
        add_edge_index = np.where(graph[:, -1] == 0.)
        grad_add = grad[add_edge_index, :]
        grad_add = np.squeeze(grad_add)
        # sort grad_add
        a = grad_add[grad_add[:, 2].argsort()]
        # a = a[::-1]
        edge = []
        for zi in a:
            if self.constraints[int(zi[0])] == 0 or self.constraints[
                int(zi[1])] == 0 or zi[-1] >= 0:  # caller cannot be contained in constraints
                continue
            else:
                edge = zi[:2].astype(np.int64)
                break
        if edge == []:
            print("zkf no edge to add")
            return graph, [-1, -1, -1]
        ii = np.where(np.all((graph[:, :2] == edge), axis=1) == True)
        graph[ii, 2] = 1
        # self.cur_graph = graph
        # print("len con:", str(len(self.constraints)), "  size graph:", str(max(graph[:, 0])), ',',
        #       str(max(graph[:, 1])),
        #       ', sen_api_idx:', str(np.max(self.sen_api_idx)))
        self.constraints[edge[1]] = -1

        return graph, [-1, int(np.squeeze(edge[0])), int(np.squeeze(edge[1]))]

    def add_edge2(self, graph):
        grad = self.get_gradient2(graph)
        a = grad[grad[:, 2].argsort()]
        edge = []
        for zi in a:
            if self.constraints[int(zi[0])] == 0 or self.constraints[
                int(zi[1])] == 0:  # caller cannot be contained in constraints
                continue
            elif zi[-1] >= 0:
                break
            else:
                pos_idc = np.all(graph[:, :2] == zi[:2].astype(int), axis=-1)
                if np.any(pos_idc) and np.squeeze(graph[pos_idc])[-1] == 0.:
                    edge = zi[:2].astype(int)
                    break
        if edge == []:
            print("zkf no edge to add")
            return graph, [-1, -1, -1]
        # ii = np.where(np.all((graph[:, :2] == edge), axis=1) == True)
        # graph[ii[0], 2] = 1

        graph[np.all((graph[:, :2] == edge), axis=1), 2] = 1

        # self.cur_graph = graph
        # print("len con:", str(len(self.constraints)), "  size graph:", str(max(graph[:, 0])), ',',
        #       str(max(graph[:, 1])),
        #       ', sen_api_idx:', str(np.max(self.sen_api_idx)))
        self.constraints[edge[1]] = -1

        return graph, [-1, int(np.squeeze(edge[0])), int(np.squeeze(edge[1]))]

    def rewiring(self, graph):
        '''
        :param graph: spare graph
        :return: modified graph
            The rewiring operation consists of two steps:
            1. find the edge with max gradient, delete edge <v1, v2>
            2. calculate the gradient of each edge from node v1 -- <v1, v3> and calculate the gradient of each edge to
            node v2 --<v3, v2> , find the node v3 and add the edge <v1,v3> and <v3, v2>
        '''
        grad = self.get_gradient(graph)
        # get the delete edge with max grad
        delete_edge_index = np.where(graph[:, -1] == 1.)[0]
        grad_add = grad[delete_edge_index, :]
        # grad_add = np.squeeze(grad_add)
        # sort grad_add, find the deletable edge
        a = grad_add[grad_add[:, 2].argsort()]
        a = a[::-1]
        del_edge = []
        for zi in a:
            if self.constraints[int(zi[0])] == 0:  # caller cannot be contained in constraints
                continue
            elif zi[-1] < 0:
                break
            else:
                del_edge = zi[:2].astype(int)
                break
        if del_edge == []:
            return graph, [-1, -1, -1]  # cannot find a edge for removing

        ii = np.where(np.all((graph[:, :2] == del_edge), axis=1) == True)[0]
        graph[ii, 2] = 0

        # get the add edge
        add_edge_ind = np.where(graph[:, -1] == 0.)[0]
        tmp_add = grad[add_edge_ind, :]
        # tmp_add = np.squeeze(tmp_add)
        beg_node = del_edge[0]
        tar_node = del_edge[1]

        # beg_idx = np.where(tmp_add[:, 0] == beg_node)[0]
        beg_grad = grad[add_edge_ind[tmp_add[:, 0] == beg_node], :]
        mid1 = grad[:, 1]

        # end_idx = np.where(tmp_add[:, 1] == tar_node)[0]
        end_grad = grad[add_edge_ind[tmp_add[:, 1] == tar_node], :]
        mid2 = grad[:, 0]

        inter = np.intersect1d(mid1, mid2)
        tt = inter.tolist()
        grad_tmp = []
        for za in tt:
            ibeg = np.where(beg_grad[:, 1] == za)[0]
            iend = np.where(end_grad[:, 0] == za)[0]
            # ibeg, iend = beg_grad[:, 1] == za, end_grad[:, 0] == za
            if ibeg.size == 0 or iend.size == 0:
                grad_tmp.append(-1)
            else:
                grad_tmp.append((beg_grad[ibeg, -1] + end_grad[iend, -1])[0])

        # sort grad_add
        grad_tmp = np.array([inter, grad_tmp]).transpose()
        a = grad_tmp[grad_tmp[:, -1].argsort()]
        # a = a[::-1]
        mid_node = -1
        for zi in a:
            if self.constraints[int(zi[0])] == 0 or \
                    beg_node == zi[0] or tar_node == zi[0]:  # caller cannot be contained in constraints
                continue
            elif zi[-1] >= 0:
                break
            else:
                mid_node = zi[0].astype(int)
                break
        # ii1 = np.where(np.all((graph[:, :2] == [beg_node, mid_node]), axis=1) == True)[0]
        graph[np.all((graph[:, :2] == [beg_node, mid_node]), axis=1), 2] = 1
        # ii2 = np.where(np.all((graph[:, :2] == [mid_node, tar_node]), axis=1) == True)[0]
        graph[np.all((graph[:, :2] == [mid_node, tar_node]), axis=1), 2] = 1
        # self.constraints[beg_node] = -1
        self.constraints[mid_node] = -1

        # print("len con:", str(len(self.constraints)), "  size graph:", str(max(graph[:, 0])), ',',
        #       str(max(graph[:, 1])),
        #       ' sen_api_idx:', str(np.max(self.sen_api_idx)))

        return graph, [beg_node, tar_node, mid_node]

    def get_gradient(self, graph):
        # calculate the grade of each edge
        triple_copy = graph.copy()
        tmpz = torch.from_numpy(triple_copy).float().to(self.device)
        triple_torch = Variable(tmpz, requires_grad=True)
        feature = MalScan.get_extra_feature(triple_torch,
                                            self.sen_api_idx,
                                            adj_size=self.adj_size,
                                            is_sp2dense=True,
                                            device=self.device)
        # feature = self.getDegreeCentrality(triple_torch, self.sen_api_idx).to(device)
        #
        # densegraph = self.to_adjmatrix(triple_torch)
        # feature_katz = self.katz_feature_torch(densegraph, self.sen_api_idx).to(device)
        # feature = torch.cat((feature, np.squeeze(feature_katz)), 0)

        feature = torch.reshape(feature, (1, -1))
        # dist = (torch.sum(feature.float() - np.squeeze(X_train.float()), 1)).pow(2)
        dist = torch.cat(
            [torch.sum((feature.float() - torch.squeeze(x.to(self.device))).pow(2), 1) for x in self.X_train])
        loss = torch.sum(self.w * (torch.sigmoid(self.steep * dist)))
        loss = torch.reshape(loss, (1, -1)).contiguous()
        loss.backward()

        print("Debug: distance loss is ", loss.detach().cpu().item())

        tmp = triple_torch.grad.data.cpu().numpy()
        grad = np.concatenate((triple_torch.cpu()[:, :2].data.numpy(), tmp[:, 2:]), 1)
        return grad

    def get_gradient2(self, graph, is_dense=False):
        # calculate the grade of each edge
        # start_time = time.time()
        graph_dense = torch.sparse_coo_tensor(graph[:, :2].T,
                                              graph[:, 2],
                                              size=(self.adj_size, self.adj_size)
                                              ).to_dense().float().to(self.device)
        graph_dense.requires_grad = True
        feature = MalScan.get_extra_feature(graph_dense,
                                            self.sen_api_idx,
                                            adj_size=self.adj_size,
                                            is_sp2dense=False,
                                            device=self.device)

        feature = torch.reshape(feature, (1, -1))
        # total_time = time.time() - start_time
        # print("cost time 1-1-1: seconds {:.4}.".format(total_time))
        dist = torch.cat(
            [torch.sum((feature.float() - torch.squeeze(x.to(self.device))).pow(2), 1) for x in self.X_train])
        # dist = torch.sum((torch.squeeze(X_train.to(self.device)) - torch.squeeze(feature.float())).pow(2.), 1)
        loss = torch.sum(self.w * (torch.sigmoid(self.steep * dist)))
        loss = torch.reshape(loss, (1, -1)).contiguous()

        print("Debug: distance loss is ", loss.detach().cpu().item())

        loss.backward()
        if not is_dense:
            tmp = graph_dense.grad.data.to_sparse()
            triple = torch.hstack([tmp.indices().t(), tmp.values()[:, None]])
            diag_indicator = triple[:, 0] == triple[:, 1]
            triple = triple[~diag_indicator]
            return triple.cpu().numpy()
        else:
            return graph_dense.grad.data

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0] * len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
