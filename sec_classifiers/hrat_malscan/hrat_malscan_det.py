import time
import numpy as np
import torch
import warnings

from hashsmooth.classifier_template import BasicClassifier


class MalScan(BasicClassifier):
    def __init__(self, train_x: (torch.Tensor, torch.utils.data.dataloader), train_y: torch.Tensor):
        self.train_x = train_x
        self.train_y = train_y

    def eval(self):
        pass

    def train(self):
        pass

    def predict(self, x: (np.ndarray, torch.Tensor), adj_size: int, top_k=1, x_sensitive_dix=None,
                batch_size=64, device='cpu', verbose=False) -> torch.Tensor:
        if isinstance(self.train_x, torch.Tensor):
            train_x = torch.split(self.train_x, batch_size)
        elif isinstance(self.train_x, torch.utils.data.dataloader.DataLoader):
            train_x = self.train_x
        else:
            raise ValueError
        if x_sensitive_dix is None:
            assert isinstance(x, torch.Tensor)
            malscan_feature = x.to(device)
        else:
            malscan_feature = MalScan.get_extra_feature(x, x_sensitive_dix,
                                                        adj_size,
                                                        device)

        dist = torch.cat(
            [torch.sum((torch.squeeze(x_batch).to(device) - torch.squeeze(malscan_feature)).pow(2.), 1)
             for x_batch in train_x])
        ind = torch.argsort(dist)
        label = self.train_y[ind[:top_k]]

        if verbose:
            label_total = self.train_y[ind]
            list_label = label_total.cpu().numpy()
            benign_idx = np.argwhere(list_label == 1)
            min_dist = dist[ind[benign_idx[0][0]]]
            print("Debug: the minimum distance to a benign file is {:.4}.".format(min_dist))

        unique_label = torch.unique(self.train_y)
        unique_label = unique_label.long()
        count = np.zeros(unique_label.shape[0])
        for i in label:
            count[unique_label[i.long()]] += 1
        ii = torch.argmax(torch.from_numpy(count))
        final_label = unique_label[ii]
        return final_label

    @staticmethod
    def get_extra_feature(x: (np.ndarray, torch.Tensor),
                          x_sensitive_dix: np.ndarray,
                          adj_size: int,
                          is_sp2dense=True,
                          device='cpu'
                          ) -> torch.Tensor:
        if is_sp2dense:
            adj = torch.sparse_coo_tensor(x[:, :2].T,
                                          x[:, 2],
                                          size=(adj_size, adj_size)
                                          ).to(device)
            adj_dense = adj.to_dense()
        else:
            adj_dense = x
        degree_fea = MalScan._degree_centrality_torch(adj_dense, x_sensitive_dix, adj_size)
        katz_fea = MalScan._katz_feature_torch(adj_dense, x_sensitive_dix, adj_size)
        return torch.cat((degree_fea, torch.squeeze(katz_fea)), 0)

    @staticmethod
    def _degree_centrality_torch(adj_dense: torch.Tensor, x_sensitive_dix: np.ndarray, adj_size: int):
        idx_matrix = np.zeros((len(x_sensitive_dix), adj_size))

        ii = np.where(x_sensitive_dix != -1)
        idx_matrix[ii, x_sensitive_dix[ii]] = 1
        idx_matrix = torch.from_numpy(idx_matrix).to(adj_dense.device)

        if adj_dense.shape[0] > len(idx_matrix):
            _sub = adj_dense.shape[0] - len(idx_matrix)
            for za in range(_sub):
                idx_matrix.append[0]

        adj_dense = torch.squeeze(adj_dense)
        all_degree = torch.div((torch.sum(adj_dense, 0) + torch.sum(adj_dense, 1)).float(),
                               float(adj_dense.shape[0] - 1))
        degree_centrality = torch.matmul(idx_matrix, all_degree.type_as(idx_matrix))
        return degree_centrality

    @staticmethod
    def _katz_feature_torch(adj_dense: torch.Tensor, x_sensitive_dix: np.ndarray, adj_size: int,
                            alpha=0.1, beta=1.0, normalized=True):
        # if not adj_dense.is_sparse:
        adj_dense = torch.squeeze(adj_dense)
        graph = adj_dense.T
        b = torch.ones((adj_size, 1), device=graph.device) * float(beta)
        A = torch.eye(adj_size, adj_size).to(graph.device).float() - (alpha * graph.float())
        L = torch.linalg.solve(A, b)
        if normalized:
            norm = torch.sign(sum(L)) * torch.norm(L)
        else:
            norm = 1.0
        centrality = torch.div(L, norm)

        idx_matrix = np.zeros((len(x_sensitive_dix), adj_size))
        ii = np.where(x_sensitive_dix != -1)
        idx_matrix[ii, x_sensitive_dix[ii]] = 1
        idx_matrix = torch.from_numpy(idx_matrix).to(graph.device)
        katz_centrality = torch.matmul(idx_matrix, centrality.type_as(idx_matrix))
        return katz_centrality
