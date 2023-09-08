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
                malscan_feature = MalScan.get_extra_feature(x, x_sensitive_dix,
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
                min_value, _ = torch.min(dist, dim=-1, keepdim=True)
                # label = self.train_y[(dist.isclose(min_value)).nonzero().squeeze(dim=-1)]
                min_v_indices = dist.isclose(min_value).nonzero()
                final_label = []
                for idx in range(malscan_feature.shape[0]):
                    l = self.train_y[min_v_indices[min_v_indices[:, 0] == idx][:, 1]]
                    pred_l = l.bincount(minlength=self.n_class).argmax().data.item()
                    final_label.append(pred_l)
                final_label = np.array(final_label)
            return final_label

    @staticmethod
    def get_extra_feature(x_batch: (np.ndarray, torch.Tensor),
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
                                            ).float().to_dense().to(device) for x in x_batch]
            adj_dense = torch.stack(adjs)
        else:
            adj_dense = x_batch if len(x_batch.shape) == 3 else x_batch[None, ...]
        degree_fea = MalScan._degree_centrality(adj_dense, x_sensitive_dix, adj_size)
        katz_fea = MalScan._katz_feature(adj_dense, x_sensitive_dix, adj_size)
        return torch.cat([degree_fea, katz_fea], -1).squeeze()

    @staticmethod
    def _degree_centrality(adj_dense: torch.Tensor, x_sensitive_dix: np.ndarray, adj_size: int):
        idx_matrix = torch.zeros((x_sensitive_dix.shape[0], adj_size), dtype=float, device=adj_dense.device)
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
    def _katz_feature(adj_dense: torch.Tensor, x_sensitive_dix: np.ndarray, adj_size: int,
                      alpha=0.1, beta=1.0, normalized=True):
        # if not adj_dense.is_sparse:
        # adj_dense = torch.squeeze(adj_dense)
        graph = adj_dense.permute([0, 2, 1])
        # bs = graph.shape[0]
        # b = torch.ones((bs, adj_size, 1), device=graph.device) * float(beta)
        # A = torch.eye(adj_size, adj_size).to(graph.device).float().repeat(bs, 1, 1) - (alpha * graph.float())
        # L = torch.linalg.solve(A, b).squeeze(-1)
        b = torch.ones((adj_size, 1), device=adj_dense.device) * float(beta)
        L = torch.stack([torch.linalg.solve(
            torch.eye(adj_size, adj_size, device=adj_dense.device).float() - alpha * adj.float().clamp(0., 1.), b).squeeze()
                         for
                         adj in graph])

        if normalized:
            norm = torch.sign(torch.sum(L, dim=-1, keepdim=True)) * torch.norm(L, dim=-1, keepdim=True)
        else:
            norm = 1.0
        centrality = torch.div(L, norm)

        idx_matrix = torch.zeros((len(x_sensitive_dix), adj_size), dtype=float, device=adj_dense.device)
        ii = np.where(x_sensitive_dix != -1)
        idx_matrix[ii[0], x_sensitive_dix[ii]] = 1.
        katz_centrality = (idx_matrix @ centrality[..., None].type_as(idx_matrix)).squeeze(-1)
        return katz_centrality.clamp(min=0)  # suppress the negative values

    @staticmethod
    def get_extra_feature_sp(x_batch: (np.ndarray, torch.Tensor),
                             x_sensitive_dix: np.ndarray,
                             adj_size: int,
                             device='cpu'
                             ) -> torch.Tensor:
        adjs_sp = torch.stack([torch.sparse_coo_tensor(x[:, :2].T,
                                                       x[:, 2],
                                                       size=(adj_size, adj_size)
                                                       ).float().to(device) for x in x_batch])

        degree_fea = MalScan._degree_centrality_sp(adjs_sp, x_sensitive_dix, adj_size)
        katz_fea = MalScan._katz_feature_sp(adjs_sp, x_sensitive_dix, adj_size)
        return torch.cat([degree_fea, katz_fea], -1).squeeze()

    @staticmethod
    def _degree_centrality_sp(adj_sp: torch.Tensor, x_sensitive_dix: np.ndarray, adj_size: int):
        idx_matrix = torch.zeros((len(x_sensitive_dix), adj_size), dtype=float, device=adj_sp.device)
        ii = np.where(x_sensitive_dix != -1)
        idx_matrix[ii[0], x_sensitive_dix[ii]] = 1.

        if adj_sp.shape[0] > len(idx_matrix):
            _sub = adj_sp.shape[0] - len(idx_matrix)
            for za in range(_sub):
                idx_matrix.append[0]

        # adj_dense = torch.squeeze(adj_dense)
        all_degree = torch.div((torch.sparse.sum(adj_sp, -2) + torch.sparse.sum(adj_sp, -1)).float(),
                               float(adj_size - 1)).to_dense()
        # degree_centrality = (idx_matrix @ all_degree[..., None].type_as(idx_matrix)).squeeze(-1)
        degree_centrality = (idx_matrix @ all_degree[..., None].type_as(idx_matrix)).squeeze(-1)
        return degree_centrality

    @staticmethod
    def _katz_feature_sp(adj_sp: torch.Tensor, x_sensitive_dix: np.ndarray, adj_size: int,
                         alpha=0.1, beta=1.0, normalized=True):
        # if not adj_dense.is_sparse:
        b = torch.ones((adj_size, 1), device=adj_sp.device) * float(beta)
        L = torch.stack([torch.linalg.solve(
            torch.eye(adj_size, adj_size, device=adj_sp.device).float() - alpha * adj.to_dense().T.float(), b).squeeze() for
             adj in adj_sp])
        # bs = adj_sp.shape[0]
        # b = torch.ones((bs, adj_size, 1), device=adj_sp.device) * float(beta)
        #
        # A_sp = adj_sp.transpose(2, 1) * (-1.0) * alpha + torch.stack(
        #     [torch.sparse_coo_tensor(np.array([np.arange(0, adj_size), np.arange(0, adj_size)]),
        #                              values=torch.tensor([1.] * adj_size), device=adj_sp.device)] * bs)
        # L = torch.linalg.solve(A_sp.to_dense(), b).squeeze(-1)
        if normalized:
            norm = torch.sign(torch.sum(L, dim=-1, keepdim=True)) * torch.norm(L, dim=-1, keepdim=True)
        else:
            norm = 1.0
        centrality = torch.div(L, norm)

        idx_matrix = torch.zeros((len(x_sensitive_dix), adj_size), dtype=float, device=adj_sp.device)
        ii = np.where(x_sensitive_dix != -1)
        idx_matrix[ii[0], x_sensitive_dix[ii]] = 1.
        katz_centrality = (idx_matrix @ centrality[..., None].type_as(idx_matrix)).squeeze(-1)
        return katz_centrality
