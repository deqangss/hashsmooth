import time
import numpy as np
import torch
import warnings

from hashsmooth.classifier_template import BasicClassifier


class MalScan(BasicClassifier):
    def __init__(self, train_x: (torch.Tensor, torch.utils.data.dataloader), train_y: torch.Tensor,
                 batch_size=64):
        super(MalScan, self).__init__(batch_size)
        self.train_x = train_x
        self.train_y = train_y
        if isinstance(self.train_x, torch.Tensor):
            self.train_x = torch.split(self.train_x, self.batch_size)
        elif isinstance(self.train_x, torch.utils.data.dataloader.DataLoader):
            self.train_x = self.train_x
        else:
            raise ValueError
        self.n_class = len(torch.unique(self.train_y))

    def eval(self):
        pass

    def train(self):
        pass

    def predict(self, x: (np.ndarray, torch.Tensor), adj_size: int, top_k=1, x_sensitive_dix=None,
                device='cpu', verbose=False) -> np.ndarray:
        with torch.no_grad():
            if x_sensitive_dix is None:
                assert isinstance(x, torch.Tensor)
                malscan_feature = x.to(device)
            else:
                malscan_feature = MalScan.get_extra_feature(x, x_sensitive_dix,
                                                            adj_size,
                                                            True,
                                                            device)

            dist = torch.cat(
                [torch.sum((torch.squeeze(x_batch).to(device) - torch.squeeze(malscan_feature)).pow(2.), 1)
                 for x_batch in self.train_x])
            if top_k > 1:
                ind = torch.argsort(dist)
                label = self.train_y[ind[:top_k]]
                warnings.warn("Do not actually implement.\n")
            else:
                min_value = torch.min(dist)
                label = self.train_y[(dist.isclose(min_value)).nonzero().squeeze(dim=-1)]

            if verbose:
                label_total = self.train_y[ind]
                list_label = label_total.cpu().numpy()
                benign_idx = np.argwhere(list_label == 1)
                min_dist = dist[ind[benign_idx[0][0]]]
                print("Debug: the minimum distance to a benign file is {:.4}.".format(min_dist))

            count_t = label.bincount()
            final_label = torch.argmax(count_t).data.item()
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
                                          ).float().to(device)
            adj_dense = adj.to_dense()
        else:
            adj_dense = x
        degree_fea = MalScan._degree_centrality_torch(adj_dense, x_sensitive_dix, adj_size)
        katz_fea = MalScan._katz_feature_torch(adj_dense, x_sensitive_dix, adj_size)
        return torch.cat((degree_fea, torch.squeeze(katz_fea)), 0)

    @staticmethod
    def _degree_centrality_torch(adj_dense: torch.Tensor, x_sensitive_dix: np.ndarray, adj_size: int):
        idx_matrix = torch.zeros((len(x_sensitive_dix), adj_size), dtype=float, device=adj_dense.device)
        ii = np.where(x_sensitive_dix != -1)
        idx_matrix[ii[0], x_sensitive_dix[ii]] = 1.

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

        idx_matrix = torch.zeros((len(x_sensitive_dix), adj_size), dtype=float, device=adj_dense.device)
        ii = np.where(x_sensitive_dix != -1)
        idx_matrix[ii[0], x_sensitive_dix[ii]] = 1.
        katz_centrality = torch.matmul(idx_matrix, centrality.type_as(idx_matrix))
        return katz_centrality

    def predict_batch(self, x: (np.ndarray, torch.Tensor), adj_size: int, top_k=1, x_sensitive_dix=None,
                      device='cpu', verbose=False) -> np.ndarray:
        """
        prediction for a batch of representaions from an instance
        :param x: a batch of representaions from an instance
        :param adj_size: number of nodes
        :param top_k: k nearest neighbour
        :param x_sensitive_dix: indices of sensitive api
        :param device: cpu or gpu
        :param verbose: print useful information
        :return: predicted label or labels
        """
        with torch.no_grad():
            if x_sensitive_dix is None:
                assert isinstance(x, torch.Tensor)
                malscan_feature = x.to(device)
            else:
                malscan_feature = MalScan.get_extra_feature_batch(x, x_sensitive_dix,
                                                                  adj_size,
                                                                  True,
                                                                  device)
            malscan_feature = malscan_feature if len(malscan_feature.shape) == 2 else malscan_feature[None, ...]
            dist = torch.cat(
                [torch.sum((malscan_feature[:, None, :] - x_batch[None, ...].to(device)).pow(2.), -1) for x_batch in
                 self.train_x], dim=1
            )
            if top_k > 1:
                ind = torch.argsort(dist)
                label = self.train_y[ind[:, :top_k]].cpu().numpy()
                warnings.warn("Do not actually implement.\n")
                count_t = np.apply_along_axis(lambda xx: np.bincount(xx, minlength=self.n_class), axis=-1, arr=label)
                final_label = np.argmax(count_t, axis=-1)
            else:
                min_value, _ = torch.min(dist, dim=-1)
                # label = self.train_y[(dist.isclose(min_value)).nonzero().squeeze(dim=-1)]
                min_v_indices = dist.isclose(min_value).nonzero()
                final_label = []
                for idx in range(malscan_feature.shape[0]):
                    l = self.train_y[min_v_indices[min_v_indices[:, 0] == idx][:, 1]]
                    pred_l = l.bincount().argmax().data.item()
                    final_label.append(pred_l)
                final_label = np.array(final_label)
            return final_label

    @staticmethod
    def get_extra_feature_batch(x_batch: (np.ndarray, torch.Tensor),
                                x_sensitive_dix: np.ndarray,
                                adj_size: int,
                                is_sp2dense=True,
                                device='cpu'
                                ) -> torch.Tensor:
        if is_sp2dense:
            x_batch = x_batch if len(x_batch.shape) == 3 else x_batch[None, ...]
            adjs = [torch.sparse_coo_tensor(x[:, :2].T,
                                            x[:, 2],
                                            size=(adj_size, adj_size)
                                            ).float().to(device).to_dense() for x in x_batch]
            adj_dense = torch.stack(adjs)
        else:
            adj_dense = x_batch if len(x_batch.shape) == 3 else x_batch[None, ...]
        degree_fea = MalScan._degree_centrality_batch(adj_dense, x_sensitive_dix, adj_size)
        katz_fea = MalScan._katz_feature_batch(adj_dense, x_sensitive_dix, adj_size)
        return torch.cat([degree_fea, katz_fea], -1)

    @staticmethod
    def _degree_centrality_batch(adj_dense: torch.Tensor, x_sensitive_dix: np.ndarray, adj_size: int):
        idx_matrix = torch.zeros((len(x_sensitive_dix), adj_size), dtype=float, device=adj_dense.device)
        ii = np.where(x_sensitive_dix != -1)
        idx_matrix[ii[0], x_sensitive_dix[ii]] = 1.

        if adj_dense.shape[0] > len(idx_matrix):
            _sub = adj_dense.shape[0] - len(idx_matrix)
            for za in range(_sub):
                idx_matrix.append[0]

        # adj_dense = torch.squeeze(adj_dense)
        all_degree = torch.div((torch.sum(adj_dense, -2) + torch.sum(adj_dense, -1)).float(),
                               float(adj_size - 1))
        degree_centrality = (idx_matrix @ all_degree[..., None].type_as(idx_matrix)).squeeze(-1)
        return degree_centrality

    @staticmethod
    def _katz_feature_batch(adj_dense: torch.Tensor, x_sensitive_dix: np.ndarray, adj_size: int,
                            alpha=0.1, beta=1.0, normalized=True):
        # if not adj_dense.is_sparse:
        # adj_dense = torch.squeeze(adj_dense)
        graph = adj_dense.permute([0, 2, 1])
        bs = graph.shape[0]
        b = torch.ones((bs, adj_size, 1), device=graph.device) * float(beta)
        A = torch.eye(adj_size, adj_size).to(graph.device).float().repeat(bs, 1, 1) - (alpha * graph.float())
        L = torch.linalg.solve(A, b).squeeze(-1)
        if normalized:
            norm = torch.sign(torch.sum(L, dim=-1, keepdim=True)) * torch.norm(L, dim=-1, keepdim=True)
        else:
            norm = 1.0
        centrality = torch.div(L, norm)

        idx_matrix = torch.zeros((len(x_sensitive_dix), adj_size), dtype=float, device=adj_dense.device)
        ii = np.where(x_sensitive_dix != -1)
        idx_matrix[ii[0], x_sensitive_dix[ii]] = 1.
        katz_centrality = (idx_matrix @ centrality[..., None].type_as(idx_matrix)).squeeze(-1)
        return katz_centrality
