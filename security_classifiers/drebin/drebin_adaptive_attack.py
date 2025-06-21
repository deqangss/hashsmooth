"""
@ARTICLE{9321695,
  author={D. {Li} and Q. {Li} and Y. F. {Ye} and S. {Xu}},
  journal={IEEE Transactions on Network Science and Engineering},
  title={A Framework for Enhancing Deep Neural Networks against Adversarial Malware},
  year={2021},
  doi={10.1109/TNSE.2021.3051354}}
"""

import torch
import torch.nn.functional as F

# from tools import utils
# logger = utils.logging.getLogger("pgdl1-attack-core")
# logger.addHandler(utils.ErrorHandler)


EXP_OVER_FLOW = 1e-30


class PGDl1(object):
    """
    Projected gradient descent (ascent) with gradients 'normalized' using l1 norm.
    By comparing BCA, the api removal is leveraged

    Parameters
    ---------
    @param injection_x, injection
    @param removal_x, removal
    @param device, 'cpu' or 'cuda'
    """

    def __init__(self, injection_x=None, removal_x=None, omega_add=None, omega_rmv=None, is_attacker=True, device=None):
        self.injection_x = injection_x
        self.removal_x = removal_x
        self.omega_add = omega_add
        self.omega_rmv = omega_rmv
        self.is_attacker = is_attacker

        self.device = device
        # self.inverse_feature = InverseDroidFeature() neglect
        self.initialize()

    def initialize(self):
        assert self.injection_x is not None
        self.injection_x = torch.LongTensor(self.injection_x).to(self.device)
        assert self.removal_x is not None
        self.removal_x = torch.LongTensor(self.removal_x).to(self.device)
        if self.omega_add is not None:
            self.omega_add = torch.LongTensor(self.omega_add).to(self.device)
        if self.omega_rmv is not None:
            self.omega_rmv = torch.LongTensor(self.omega_rmv).to(self.device)

    def perturb(self, model, x,  label=None, steps=10, verbose=True):
        """
        perturb node feature vectors

        Parameters
        -----------
        @param model, a victim model
        @param x: torch.FloatTensor, node feature vectors (each represents the occurrences of apis in a graph) with shape [batch_size, number_of_graphs, vocab_dim]
        @param label: torch.LongTensor, ground truth labels
        @param steps: Integer, maximum number of perturbations
        """
        if x is None or x.shape[0] <= 0:
            return []
        adv_x = x
        worst_x = x.detach().clone()
        # self.padding_mask = torch.sum(adv_x, dim=-1, keepdim=True) > 1  # we set a graph contains two apis at least
        model.eval()
        pred_y = model.predict(adv_x)
        if hasattr(model, 'ABSTAIN'):
            done = torch.logical_and(pred_y != label, pred_y != model.ABSTAIN)
        else:
            done = pred_y != label

        worst_x[done] = adv_x[done]

        for t in range(steps):
            var_adv_x = torch.autograd.Variable(adv_x, requires_grad=True)
            loss = model.get_loss(var_adv_x, label)
            print("loss: ", loss)
            grad = torch.autograd.grad(loss, var_adv_x)[0]
            perturbation, direction = self.get_perturbation(grad, x, adv_x)
            # stop perturbing the examples that are successful to evade the victim
            perturbation[done] = 0.
            adv_x = torch.clamp(adv_x + perturbation * direction, min=0., max=1.)

            pred_y = model.predict(adv_x)
            if hasattr(model, 'ABSTAIN'):
                new_done = torch.logical_and(pred_y != label, pred_y != model.ABSTAIN)
                done = torch.logical_or(done, new_done)
            else:
                new_done = pred_y != label
                done = torch.logical_or(done, new_done)
            worst_x[done] = adv_x[done]

            if verbose:
                print("Attack step {} with accuracy {:.2f}%.\n".format(t+1, 100 - done.sum().item()/float(len(done)) * 100))
            if torch.all(done):
                break
        worst_x[~done] = adv_x[~done]
        return worst_x

    def get_perturbation(self, gradients, features, adv_features):
        # 1. mask paddings
        # gradients = gradients * self.padding_mask

        # 2. look for allowable position, because only '1--> -' and '0 --> +' are permitted
        #    2.1 insertion
        pos_insertion = (adv_features <= 0.5) * 1
        grad4insertion = (gradients > 0) * (pos_insertion & self.injection_x) * gradients
        #    2.2 api removal
        pos_removal = (adv_features > 0.5) * 1
        grad4removal = (gradients <= 0) * (pos_removal & self.removal_x) * gradients
        if self.is_attacker and (self.omega_add is not None) and (self.omega_rmv is not None):
            # 2.2.1 cope with the interdependent apis
            check_nonexist_insert = (pos_insertion[:, None, :] ^ self.omega_add) & self.omega_add
            grad4insertion += torch.sum(gradients[:, None, :] * check_nonexist_insert, dim=-1)

            checking_exist_removal = pos_removal[:, None, :] & self.omega_rmv
            grad4removal += torch.sum(gradients * checking_exist_removal, dim=-1)

        gradients = grad4removal + grad4insertion

        # 3. remove duplications
        un_mod = torch.abs(features - adv_features) <= 1e-6
        gradients = gradients * un_mod

        # 4. look for important position
        absolute_grad = torch.abs(gradients).reshape(features.shape[0], -1)
        _, position = torch.max(absolute_grad, dim=-1)
        perturbations = F.one_hot(position, num_classes=absolute_grad.shape[-1])
        perturbations = perturbations.reshape(features.shape)
        directions = torch.sign(gradients) * (perturbations > 1e-6)
        # directions = torch.where(directions == 0., -1., directions)

        # 5. tailor the interdependent apis
        if self.is_attacker and (self.omega_add is not None) and (self.omega_rmv is not None):
            iterdep_feat_flag = torch.sum(self.omega_add, dim=-1) > 0
            perturbations += (torch.any(directions[:, iterdep_feat_flag] > 0, dim=-1, keepdim=True)) * check_nonexist_insert
            directions += perturbations * self.omega_add

            iterdep_rmv_flag = torch.sum(self.omega_rmv, dim=-1) > 0
            perturbations += (torch.any(directions[:, iterdep_rmv_flag] < 0, dim=-1, keepdim=True)) * checking_exist_removal
            directions += perturbations * self.omega_rmv
            
        return perturbations, directions

