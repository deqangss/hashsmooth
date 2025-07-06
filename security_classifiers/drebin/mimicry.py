"""

"""
import torch
import numpy as np
import torch.nn.functional as F
from model import RandomSmooth4Drebin, HashSmooth4Drebin, SparsitySmooth4Drebin

EXP_OVER_FLOW = 1e-120


class Mimicry(object):
    """
    Mimicry attack: inject the graph of benign file into malicious ones

    Parameters
    ---------
    @param ben_x: torch.FloatTensor, feature vectors with shape [number_of_benign_files, vocab_dim]
    @param device, 'cpu' or 'cuda'
    """

    def __init__(self, ben_x, injection_x=None, removal_x=None, device=None):
        self.injection_x = injection_x
        self.removal_x = removal_x
        self.ben_x = ben_x
        self.device = device
        self.initialize()

    def initialize(self):
        assert self.injection_x is not None
        self.injection_x = torch.LongTensor(self.injection_x).to(self.device)
        assert self.removal_x is not None
        self.removal_x = torch.LongTensor(self.removal_x).to(self.device)

    def perturb(self, model, x, label=None, trials=10, valid_dim=10000, seed=0):
        """
        modify feature vectors of malicious apps

        Parameters
        -----------
        @param model, a victim model
        @param x: torch.FloatTensor, feature vectors with shape [batch_size, vocab_dim]
        @param trials: Integer, repetition times
        @param valid_dim: Integer, the dimension of feature vectors to perturb
        @param seed: Integer, random seed
        @param is_apk: Boolean, whether produce apks
        @param verbose: Boolean, whether present attack information or not
        """
        assert trials > 0
        if x is None or len(x) <= 0:
            return []
        if len(self.ben_x) <= 0:
            return x
        if isinstance(model, HashSmooth4Drebin) and self.injection_x is not None and x.shape[-1] > \
                self.injection_x.shape[-1]:
            dim_remain = x.shape[-1] - self.injection_x.shape[-1]
            self.injection_x = F.pad(self.injection_x, (0, dim_remain)) if self.injection_x is not None else None
            self.removal_x = F.pad(self.removal_x, (0, dim_remain)) if self.removal_x is not None else None
        trials = trials if trials < len(self.ben_x) else len(self.ben_x)
        success_flag = np.array([])
        with torch.no_grad():
            torch.manual_seed(seed)
            x_mod_list = []
            y_pred_batch = model.predict(x)
            for _i, _x in enumerate(x):
                indices = torch.randperm(len(self.ben_x))[:trials]
                trial_vectors = self.ben_x[indices]
                # _x_fixed_one = ((1. - self.manipulation_x).float() * _x)[None, :]
                # modified_x = torch.clamp(_x_fixed_one + trial_vectors, min=0., max=1.)
                # modified_x, y = utils.to_tensor(modified_x.double(), torch.ones(trials,).long(), model.device)
                _x_fixed_one = torch.clamp(trial_vectors + _x, min=torch.zeros_like(trial_vectors, device=self.device), max=trial_vectors)
                inj_pos = ((_x_fixed_one - _x) > 0) * self.injection_x
                rmv_pos = ((_x_fixed_one - _x) < 0) * self.removal_x
                modified_x = _x + inj_pos.to(torch.float) - rmv_pos.to(torch.float)
                y_pred = model.predict(modified_x)
                # if hasattr(model, 'indicator') and (not self.oblivion):
                #     attack_flag = (y_pred == 0) & (model.indicator(x_density, y_pred))
                # else:
                #     attack_flag = (y_pred == 0)
                if hasattr(model, 'ABSTAIN'):
                    attack_flag = torch.logical_and(y_pred != label[_i], y_pred != model.ABSTAIN)
                else:
                    attack_flag = y_pred != label[_i]
                ben_id_sel = np.argmax(attack_flag.cpu().numpy())

                # check the attack effectiveness
                use_flag = attack_flag

                if  y_pred_batch[_i] != label[_i]:
                    success_flag = np.append(success_flag, [False])
                    rtn_adv_x = _x
                elif not use_flag[ben_id_sel]:
                    success_flag = np.append(success_flag, [False])
                    rtn_adv_x = modified_x[ben_id_sel]
                else:
                    success_flag = np.append(success_flag, [True])
                    distance = torch.abs(torch.sum(modified_x[use_flag] - _x, dim=-1))
                    min_dist_id = torch.argmin(distance)
                    rtn_adv_x = modified_x[use_flag][min_dist_id]

                x_mod_list.append(rtn_adv_x)
            return success_flag, torch.vstack(x_mod_list)
