import os
import sys
import math
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from statsmodels.stats.proportion import proportion_confint, binom_test

import drebin_utils
logger = drebin_utils.logging.getLogger("drebin-library")
logger.addHandler(drebin_utils.ErrorHandler)
binom_test = np.vectorize(binom_test)

sys.path.append('../../hashsmooth')
sys.path.append('../../randomsmooth')
sys.path.append('../../torchware')

from randomsmooth.random_tran import RandomTransformer
from sparse_smoothing.utils import binary_perturb
from sparse_smoothing.cert import binary_certificate
from hashsmooth import JaccardLSHTransformer
from hashsmooth.core import HashSmoothBase, LSHTransformer
from hashsmooth.utils_hash import lower_confidence_interval, upper_confidence_interval

ABSTAIN = -1

class DrebinSVM(nn.Module):
    def __init__(self, input_dim: int, num_classes=2, batch_size=64, model_save_dir=''):
        super(DrebinSVM, self).__init__()
        self.input_dim = input_dim
        assert num_classes == 1
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.model_save_dir = model_save_dir
        drebin_utils.mkdir(self.model_save_dir)
        self.model_save_path = os.path.join(self.model_save_dir, 'model.ckpt')

        self.model = torch.nn.Linear(self.input_dim, 1)
        self.criterion = torch.nn.BCEWithLogitsLoss()  # better than hinge loss

    def forward(self, x):
        return self.model(x)

    def fit(self, train_x_y: torch.utils.data.dataloader,
            validation_x_y: torch.utils.data.dataloader,
            epochs=1000, learning_rate=0.001, penaty_factor=0.01, device='cpu', verbose=False):
        nbatches = len(train_x_y)
        # optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        best_avg_f1 = 0.
        for i in range(epochs):
            self.model.train()
            losses, accuraies = [], []
            for i_batch, (x_train, y_train) in enumerate(train_x_y):
                x_train, y_train = x_train.to(device), y_train.to(device)
                optimizer.zero_grad()
                logits = self.model(x_train)
                loss_train = self.criterion(logits.view(-1), y_train.float())
                # weight = self.model.weight.squeeze()
                # loss_train += penaty_factor * torch.sum(weight * weight)
                loss_train.backward()
                optimizer.step()
                accuray_train = ((logits.data > 0.).view(-1) == y_train).sum().item() / x_train.size()[0]
                accuraies.append(accuray_train)
                losses.append(loss_train)

                if verbose:
                    logger.info(
                        f'Mini batch: {i * nbatches + i_batch + 1}/{epochs * nbatches} | Training loss (batch level): {losses[-1]:.4f} | Train accuracy: {accuray_train * 100:.2f}%.')

            self.model.eval()
            avg_f1_val = []
            with torch.no_grad():
                for x_val, y_val in validation_x_y:
                    x_val, y_val = x_val.to(device), y_val.to(device)
                    logits = self.model(x_val)
                    # acc_val = ((logits.data > 0.).view(-1) == y_val).sum().item()
                    y_pred = (logits.data > 0.).view(-1)
                    f1_val = f1_score(y_val.cpu().numpy(), y_pred.cpu().numpy(), average='macro')
                    avg_f1_val.append(f1_val)
                avg_f1_val = np.mean(avg_f1_val)

                if avg_f1_val >= best_avg_f1:
                    best_avg_f1 = avg_f1_val
                    best_epoch = i
                    torch.save(self.model.state_dict(), self.model_save_path)
                    logger.info(f'Model saved at path: {self.model_save_path}')

                logger.info(
                    f'Validation accuracy: {avg_f1_val * 100:.2f}% | The best validation accuracy: {best_avg_f1 * 100:.2f}% at epoch: {best_epoch}.')

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        logits = self.model(x)
        x_preds = (logits.data > 0.).view(-1).to(torch.int)
        return x_preds

    def load_model(self):
        self.model.load_state_dict(torch.load(self.model_save_path))

    def get_loss(self, x: torch.Tensor, label: int):
        logits = self.model(x)
        return self.criterion(logits.view(-1), label.float())

    def get_confidence(self, x: torch.Tensor):
        logits = self.model(x)
        return torch.sigmoid(logits)

    def certify(self, x: np.ndarray, label: (int, np.ndarray)):
        raise NotImplementedError

    def eval(self):
        self.model.eval()


class DrebinNN(nn.Module):
    def __init__(self, input_dim: int, num_classes=2, batch_size=64, model_save_dir=''):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.model_save_dir = model_save_dir
        drebin_utils.mkdir(self.model_save_dir)
        self.model_save_path = os.path.join(self.model_save_dir, 'model.ckpt')

        # manually set the parameters in NN
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, 200),
            torch.nn.ReLU(),
            torch.nn.Linear(200, 200),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.6),
            torch.nn.Linear(200, self.num_classes)
        )

        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def fit(self, train_x_y: torch.utils.data.dataloader,
            validation_x_y: torch.utils.data.dataloader,
            epochs=1000, learning_rate=0.001, device='cpu', verbose=False):
        nbatches = len(train_x_y)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        best_avg_f1 = 0.
        for i in range(epochs):
            self.model.train()
            losses, accuracies = [], []
            for i_batch, (x_train, y_train) in enumerate(train_x_y):
                x_train, y_train = x_train.to(device), y_train.to(device)
                optimizer.zero_grad()
                logits = self.model(x_train)
                loss_train = self.criterion(logits, y_train.to(torch.long))
                loss_train.backward()
                optimizer.step()
                accuray_train = (logits.argmax(dim=-1) == y_train).sum().item() / x_train.shape[0]
                accuracies.append(accuray_train)
                losses.append(loss_train)

                if verbose:
                    logger.info(
                        f'Mini batch: {i * nbatches + i_batch + 1}/{epochs * nbatches} | Training loss (batch level): {losses[-1]:.4f} | Train accuracy: {accuray_train * 100:.2f}%.')

            self.model.eval()
            avg_f1_val = []
            with torch.no_grad():
                for x_val, y_val in validation_x_y:
                    x_val, y_val = x_val.to(device), y_val.to(device)
                    logits = self.model(x_val)
                    # acc_val = (logits.argmax(dim=-1) == y_val).sum().item()
                    # acc_val /= x_val.size()[0]
                    y_pred = logits.argmax(dim=-1)
                    f1_val = f1_score(y_val.cpu().numpy(), y_pred.cpu().numpy(), average='macro')
                    avg_f1_val.append(f1_val)
                avg_f1_val = np.mean(avg_f1_val)

                if avg_f1_val >= best_avg_f1:
                    best_avg_f1 = avg_f1_val
                    best_epoch = i
                    torch.save(self.model.state_dict(), self.model_save_path)
                    logger.info(f'Model saved at path: {self.model_save_path}')

                logger.info(
                    f'Validation accuracy: {avg_f1_val * 100:.2f}% | The best validation accuracy: {best_avg_f1 * 100:.2f}% at epoch: {best_epoch}.')

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        logits = self.model(x)
        x_preds = logits.argmax(dim=-1).to(torch.int)
        return x_preds

    def load_model(self):
        self.model.load_state_dict(torch.load(self.model_save_path))

    def get_loss(self, x: torch.Tensor, label: int):
        logits = self.model(x)
        return self.criterion(logits, label.to(torch.long))

    def get_confidence(self, x: torch.Tensor):
        return torch.softmax(self.model(x), dim=-1)

    def certify(self, x: np.ndarray, label: (int, np.ndarray)):
        raise NotImplementedError

    def eval(self):
        self.model.eval()


class RandomSmooth(object):
    ABSTAIN = ABSTAIN
    def __init__(self, num_of_classes, transform_method, max_k=1000, default_mode=True):
        self.num_of_classes = num_of_classes
        self.default_mode = default_mode
        #todo: check max_k is used in transformations
        self.max_k = max_k
        self.transform_method = transform_method

    def transform(self, x):
        return self.transform_method.transform(x)

    @staticmethod
    def lc_bound(k, n, alpha):
        return proportion_confint(k, n, alpha=2 * alpha, method="beta")[0]

    @staticmethod
    def calc_radius(scores_of_true, size, keep):
        count = scores_of_true.shape[0]
        done = torch.zeros(count, dtype=torch.uint8)
        radii = torch.zeros(count, dtype=torch.long)
        radius = 0
        lhs = (1.5 - scores_of_true).squeeze(1)
        while (done.sum() < count):
            rhs = math.factorial(size - radius) * math.factorial(size - keep) / (
                    math.factorial(size) * math.factorial(size - keep - radius))
            done[torch.tensor(lhs >= rhs)] = 1
            radii[torch.tensor(lhs < rhs)] = radius
            radius += 1
        return radii

class RandomSmooth4Drebin(RandomSmooth):
    def __init__(self, clf_model, num_of_classes: int, transform_method: RandomTransformer, max_k: int,
                 default_mode: bool, model_save_dir=''):
        """
        Customized randomsmooth w.r.t. drebin
        :param clf_model: classification model
        :param num_of_classes: number of classes
        :param transform_method: input transformation model
        :param max_k: number of maximum elements
        :param default_mode: miss-classification threshold is 0.5 or not
        """
        RandomSmooth.__init__(self, num_of_classes, transform_method, max_k, default_mode)
        self.clf_model = clf_model
        self.model_save_dir = model_save_dir
        if not os.path.exists(self.model_save_dir):
            drebin_utils.mkdir(self.model_save_dir)
        self.model_save_path = os.path.join(self.model_save_dir,
                                            'model_random_{}.ckpt'.format(self.transform_method.k_randomcode))

    def eval(self):
        self.clf_model.eval()

    def get_output(self, logits: torch.Tensor):
        if isinstance(self.clf_model, DrebinSVM):
            return (logits.data > 0.).view(-1)
        elif isinstance(self.clf_model, DrebinNN):
            return logits.argmax(dim=-1)
        else:
            raise TypeError("Expect 'DrebinSVM' or 'DrebinNN'.")

    def fit(self, train_x_y: torch.utils.data.dataloader,
            validation_x_y: torch.utils.data.dataloader,
            epochs=100, learning_rate=0.001, penaty_factor=0.01, n_sampling=100, device='cpu', verbose=False):
        nbatches = len(train_x_y)
        optimizer = torch.optim.Adam(self.clf_model.parameters(), lr=learning_rate)
        best_avg_f1 = 0.
        best_epoch = 0
        for i in range(epochs):
            self.clf_model.train()
            losses, accuracies = [], []
            for i_batch, (x_train, y_train) in enumerate(train_x_y):
                x_train, y_train = x_train.to(device), y_train.to(device)
                optimizer.zero_grad()
                x_train_mask = self.transform(x_train)
                loss_train = self.clf_model.get_loss(x_train_mask, y_train)
                # decline the training convergence
                # if isinstance(self.base_classifier, DrebinSVM):
                #     weight = self.base_classifier.model.weight.squeeze()
                #     loss_train += penaty_factor * torch.sum(weight * weight)
                with torch.no_grad():
                    logits = self.clf_model(x_train_mask)
                    accuracy_train = (self.get_output(logits) == y_train).sum().item() / x_train.size()[0]
                    accuracies.append(accuracy_train)
                loss_train.backward()
                optimizer.step()
                losses.append(loss_train)
                if verbose:
                    logger.info(
                        f'Mini batch: {i * nbatches + i_batch + 1}/{epochs * nbatches} | Training loss (batch level): {losses[-1]:.4f} | Train accuracy: {accuracy_train * 100:.2f}%.')

            self.clf_model.eval()
            avg_f1_val = []
            with torch.no_grad():
                for x_val, y_val in validation_x_y:
                    x_val, y_val = x_val.to(device), y_val.to(device)
                    y_votes = self.sample_funcs(x_val, n_sampling)
                    y_pred = y_votes.argmax(dim=-1)
                    f1_val = f1_score(y_val.cpu().numpy(), y_pred.cpu().numpy(), average='macro')
                    avg_f1_val.append(f1_val)
            avg_f1_val = np.mean(avg_f1_val)

            if avg_f1_val >= best_avg_f1:
                best_avg_f1 = avg_f1_val
                best_epoch = i
                torch.save(self.clf_model.state_dict(), self.model_save_path)
                logger.info(f'Model saved at path: {self.model_save_path}')

            logger.info(
                f'Validation accuracy: {avg_f1_val * 100:.2f}% | The best validation accuracy: {best_avg_f1 * 100:.2f}% at epoch: {best_epoch}.')

    def predict(self, x: torch.Tensor, n_sampling: int, alpha: float = 0.05):
        self.eval()
        y_votes = self.sample_funcs(x, n_sampling)

        top2 = y_votes.argsort(dim=-1, descending=True)[:, :2]
        count1 = (y_votes[range(x.shape[0]), top2[:, 0]]).cpu().numpy().astype(int)
        count2 = (y_votes[range(x.shape[0]), top2[:, 1]]).cpu().numpy().astype(int)

        pred = top2[:, 0]
        abstain_flag = binom_test(count1, count1 + count2, prop=0.5) > alpha
        pred[abstain_flag] = self.ABSTAIN
        return pred

    def get_confidence(self, x: torch.Tensor, n_sampling: int):
        y_votes = self.sample_funcs(x, n_sampling)
        return torch.softmax(y_votes.float(), dim=-1)

    def load_model(self):
        self.clf_model.load_state_dict(torch.load(self.model_save_path))

    def get_loss(self, x: torch.Tensor, label: int, n, batch_size=64):
        loss = 0.
        for i in range(n):
            mask_x = self.transform(x)
            loss += self.clf_model.get_loss(mask_x, label)
        return loss

    def certify(self, x: torch.Tensor, labels: torch.Tensor, n_selection: int, n_estimation: int,
                alpha: float,
                device='cpu'):
        with torch.no_grad():
            counts_selection = self.sample_funcs(x, n_selection)
            counts_estimation = self.sample_funcs(x, n_estimation)

            c_pred = counts_selection.argmax(dim=-1).cpu().numpy()
            n_targeted = counts_estimation[range(len(c_pred)), c_pred]
            prob_underlined = lower_confidence_interval(n_targeted.cpu().numpy(), n_estimation, alpha)
            n_runnerup = counts_estimation[range(len(c_pred)), 1 - c_pred]
            prob_overlined = upper_confidence_interval(n_runnerup.cpu().numpy(), n_estimation, alpha)

            radius = np.zeros_like(c_pred, dtype=object)
            abstain_indicator = prob_underlined <= prob_overlined
            radius[abstain_indicator] = self.ABSTAIN
            incorrect_indicator = (c_pred != labels.cpu().numpy())
            incorrect_indicator_true = incorrect_indicator == True
            if np.any(incorrect_indicator_true):
                incorrect_indicator[incorrect_indicator_true] = incorrect_indicator[incorrect_indicator_true] ^ \
                                                                abstain_indicator[incorrect_indicator_true]
                radius[incorrect_indicator] = self.ABSTAIN - 1
            total_abstain_indicator = abstain_indicator | incorrect_indicator
            if np.any(~total_abstain_indicator):
                radius[~total_abstain_indicator] = self.calc_radius(
                    np.array(prob_underlined[~total_abstain_indicator])[..., None],
                    x[0].cpu().numpy().size,
                    self.max_k)
            return radius, total_abstain_indicator

    def sample_funcs(self, x: torch.Tensor, n: int):
        votes = torch.zeros((x.shape[0], self.num_of_classes), dtype=torch.long, device=x.device)
        with torch.no_grad():
            if x.shape[0] > 1:
                for i in range(n):
                    mask_x = self.transform(x)
                    logits = self.clf_model(mask_x)
                    pred_batch = self.get_output(logits).to(torch.int64)
                    votes += torch.nn.functional.one_hot(pred_batch, self.num_of_classes)
            else:
                batch_size = 64
                for idx in range(n // batch_size + 1):
                    current_batch_size = min(batch_size, n)
                    n -= current_batch_size
                    if current_batch_size <= 0:
                        break
                    mask_x = self.transform(torch.tile(x, (current_batch_size, 1)))
                    logits = self.clf_model.model(mask_x)
                    pred_batch = self.get_output(logits).to(torch.int64)
                    votes += torch.sum(torch.nn.functional.one_hot(pred_batch, self.num_of_classes), dim=0,
                                       keepdim=True)

        return votes


class SparsitySmooth(object):
    ABSTAIN = ABSTAIN
    def __init__(self, num_of_classes, pf_minus=0.8, pf_plus=0.1, default_mode=True):
        self.num_of_classes = num_of_classes
        self.pf_minus = pf_minus
        self.pf_plus = pf_plus
        self.default_mode = default_mode

    def transform(self, x):
        return binary_perturb(x, self.pf_minus, self.pf_plus)

    def calc_radius(self, votes, pre_votes, n_estimation, alpha=0.05):
        return binary_certificate(votes, pre_votes, n_estimation, alpha, self.pf_plus, self.pf_minus)


class SparsitySmooth4Drebin(SparsitySmooth):
    def __init__(self,
                 clf_model,
                 num_of_classes: int,
                 pf_minus=0.8,
                 pf_plus=0.1,
                 default_mode: bool = True,
                 model_save_dir=''
                 ):
        """
        Customized Sparse smoothing w.r.t. Drebin
        """
        SparsitySmooth.__init__(self,
                                num_of_classes,
                                pf_minus,
                                pf_plus,
                                default_mode=default_mode)
        self.clf_model = clf_model
        self.model_save_dir = model_save_dir
        if not os.path.exists(self.model_save_dir):
            drebin_utils.mkdir(self.model_save_dir)
        self.model_save_path = os.path.join(self.model_save_dir, 'model_sparsity_{}_{}.ckpt'.format(self.pf_minus, self.pf_plus))

    def eval(self):
        self.clf_model.eval()

    def fit(self, train_x_y: torch.utils.data.dataloader, validation_x_y: torch.utils.data.dataloader,
            epochs=1000, learning_rate=0.001, n_sampling=100, device='cpu', verbose=False):
        nbatches = len(train_x_y)
        optimizer = torch.optim.Adam(self.clf_model.parameters(), lr=learning_rate)
        best_avg_f1 = 0.
        for i in range(epochs):
            self.clf_model.train()
            losses, accuracies = [], []
            for i_batch, (x_train, y_train) in enumerate(train_x_y):
                x_train, y_train = x_train.to(device), y_train.to(device)

                # transformation
                x_train_mask = self.transform(x_train)

                # learning
                optimizer.zero_grad()
                loss_train = self.clf_model.get_loss(x_train_mask, y_train)
                loss_train.backward()
                optimizer.step()
                losses.append(loss_train)
                with torch.no_grad():
                    logits = self.clf_model(x_train_mask)
                    accuracy_train = (self.get_output(logits) == y_train).sum().item() / x_train.size()[0]
                    accuracies.append(accuracy_train)
                if verbose:
                    logger.info(
                        f'Mini batch: {i * nbatches + i_batch + 1}/{epochs * nbatches} | Training loss (batch level): {losses[-1]:.4f} | Train accuracy: {accuracy_train * 100:.2f}%.')

            self.clf_model.eval()
            avg_f1_val = []
            with torch.no_grad():
                for x_val, y_val in validation_x_y:
                    x_val, y_val = x_val.to(device), y_val.to(device)

                    y_votes = self.sample_funcs(x_val, n_sampling)
                    y_pred = y_votes.argmax(dim=-1)
                    f1_val = f1_score(y_val.cpu().numpy(), y_pred.cpu().numpy(), average='macro')
                    avg_f1_val.append(f1_val)
            avg_f1_val = np.mean(avg_f1_val)

            if avg_f1_val >= best_avg_f1:
                best_avg_f1 = avg_f1_val
                best_epoch = i
                torch.save(self.clf_model.state_dict(), self.model_save_path)
                logger.info(f'Model saved at path: {self.model_save_path}')

            logger.info(
                f'Validation accuracy: {avg_f1_val * 100:.2f}% | The best validation accuracy: {best_avg_f1 * 100:.2f}% at epoch: {best_epoch}.')

    def predict(self, x: torch.Tensor, n_sampling: int, alpha: float):
        self.eval()
        y_votes = self.sample_funcs(x, n_sampling)

        top2 = y_votes.argsort(dim=-1, descending=True)[:, :2]
        count1 = (y_votes[range(x.shape[0]), top2[:, 0]]).cpu().numpy()
        count2 = (y_votes[range(x.shape[0]), top2[:, 1]]).cpu().numpy()

        pred = top2[:, 0]
        prob_underlined = lower_confidence_interval(count1, n_sampling, alpha)
        if self.default_mode:
            abstain_indicator = prob_underlined < 0.5
        else:
            prob_upperlined = upper_confidence_interval(count2, n_sampling, alpha)
            abstain_indicator = prob_underlined <= prob_upperlined
        # abstain_flag = binom_test(count1, count1 + count2, prop=0.5) > alpha
        pred[abstain_indicator] = RandomSmooth.ABSTAIN
        return pred

    def load_model(self):
        self.clf_model.load_state_dict(torch.load(self.model_save_path))

    def get_loss(self, x: torch.Tensor, label: int, n, batch_size=64):
        loss = 0.
        for i in range(n):
            mask_x = self.transform(x)
            loss += self.clf_model.get_loss(mask_x, label)
        return loss

    def get_confidence(self, x: torch.Tensor, n_sampling: int):
        y_votes = self.sample_funcs(x, n_sampling)
        return torch.softmax(y_votes.float(), dim=-1)

    def certify(self, x: torch.Tensor, labels: np.ndarray, n_selection: int, n_estimation: int,
                alpha: float,
                device='cpu'):
        assert n_selection > 0 and n_estimation > 0 and 0 < alpha < 1

        counts_selection = self.sample_funcs(x, n_selection)
        counts_estimation = self.sample_funcs(x, n_estimation)

        c_pred = counts_selection.argmax(dim=-1).cpu().numpy()
        n_targeted = counts_estimation[range(len(c_pred)), c_pred]
        prob_underlined = lower_confidence_interval(n_targeted.cpu().numpy(), n_estimation, alpha)
        n_runnerup = counts_estimation[range(len(c_pred)), 1 - c_pred]
        prob_overlined = upper_confidence_interval(n_runnerup.cpu().numpy(), n_estimation, alpha)

        radius_ad = np.zeros_like(c_pred, dtype=object)
        radius_rd = np.zeros_like(c_pred, dtype=object)
        abstain_indicator = prob_underlined <= prob_overlined
        radius_ad[abstain_indicator] = self.ABSTAIN
        radius_rd[abstain_indicator] = self.ABSTAIN
        incorrect_indicator = (c_pred != labels.cpu().numpy())
        incorrect_indicator_true = incorrect_indicator == True
        if np.any(incorrect_indicator_true):
            incorrect_indicator[incorrect_indicator_true] = incorrect_indicator[incorrect_indicator_true] ^ \
                                                            abstain_indicator[incorrect_indicator_true]
            radius_ad[incorrect_indicator] = self.ABSTAIN - 1
            radius_rd[incorrect_indicator] = self.ABSTAIN - 1
        total_abstain_indicator = abstain_indicator | incorrect_indicator
        if np.any(~total_abstain_indicator):
            grid_base, grid_lower, grid_upper = self.calc_radius(
                counts_estimation[~total_abstain_indicator].cpu().numpy(),
                counts_selection[~total_abstain_indicator].cpu().numpy(),
                n_estimation,
                alpha
            )
            max_ra_base = (grid_base > 0.5)[:, :, 0].argmin(1)
            max_rd_base = (grid_base > 0.5)[:, 0, :].argmin(1)
            max_ra_loup = (grid_lower >= grid_upper)[:, :, 0].argmin(1)
            max_rd_loup = (grid_lower >= grid_upper)[:, 0, :].argmin(1)

            radius_ad[~total_abstain_indicator] = max_ra_loup
            radius_rd[~total_abstain_indicator] = max_rd_loup

        return radius_ad, radius_rd, total_abstain_indicator

    def sample_funcs(self, x: torch.Tensor, n):
        votes = torch.zeros((x.shape[0], self.num_of_classes), dtype=torch.long, device=x.device)
        with torch.no_grad():
            if x.shape[0] > 1:
                for i in range(n):
                    x_train_mask = self.transform(x)
                    logits = self.clf_model(x_train_mask)
                    pred_batch = self.get_output(logits).to(torch.int64)
                    votes += torch.nn.functional.one_hot(pred_batch, self.num_of_classes)
            else:
                batch_size = 64
                for idx in range(n // batch_size + 1):
                    current_batch_size = min(batch_size, n)
                    n -= current_batch_size
                    if current_batch_size <= 0:
                        break
                    mask_x = self.transform(torch.tile(x, (current_batch_size, 1)))
                    logits = self.clf_model(mask_x)
                    pred_batch = self.get_output(logits).to(torch.int64)
                    votes += torch.sum(torch.nn.functional.one_hot(pred_batch, self.num_of_classes), dim=0,
                                       keepdim=True)
        return votes

    def get_output(self, logits: torch.Tensor):
        if isinstance(self.clf_model, DrebinSVM):
            return (logits.data > 0.).view(-1)
        elif isinstance(self.clf_model, DrebinNN):
            return logits.argmax(dim=-1)
        else:
            raise TypeError("Expect 'DrebinSVM' or 'DrebinNN'.")


class HashSmooth(HashSmoothBase):
    ABSTAIN = ABSTAIN

    def __init__(self, num_of_classes, transform_method, max_k=1000, input_dim=1000, default_mode=True):
        super(HashSmooth, self).__init__(num_of_classes, transform_method, max_k, None, None, default_mode)
        self.input_dim = input_dim
        self.transform_method = transform_method

    def transform(self, x):
        return self.transform_method.transform(x)

    def calc_radius(self, probas, second_probas, k_hashcode=None, max_radius=None, n_grid=100):
        return self._calc_radius(probas, second_probas, k_hashcode, max_radius, n_grid)


class HashSmooth4Drebin(HashSmooth):
    def __init__(self,
                 clf_model,
                 num_of_classes: int,
                 hash_method: LSHTransformer,
                 max_k,
                 input_dim,
                 default_mode: bool,
                 model_save_dir=''
                 ):
        """
        Customized HashSmooth w.r.t. Drebin
        """
        HashSmooth.__init__(self,
                            num_of_classes,
                            hash_method,
                            max_k=max_k,
                            input_dim=input_dim,
                            default_mode=default_mode)
        self.clf_model = clf_model
        self.model_save_dir = model_save_dir
        if not os.path.exists(self.model_save_dir):
            drebin_utils.mkdir(self.model_save_dir)
        self.model_save_path = os.path.join(self.model_save_dir, 'model_hash_{}.ckpt'.format(self.k_hashcode))

    def eval(self):
        self.clf_model.eval()

    def fit(self, train_x_y: torch.utils.data.dataloader, validation_x_y: torch.utils.data.dataloader,
            epochs=1000, learning_rate=0.001, n_sampling=100, device='cpu', verbose=False):
        nbatches = len(train_x_y)
        optimizer = torch.optim.Adam(self.clf_model.parameters(), lr=learning_rate)
        best_avg_f1 = 0.
        for i in range(epochs):
            self.clf_model.train()
            losses, accuracies = [], []
            for i_batch, (x_train, y_train) in enumerate(train_x_y):
                x_train, y_train = x_train.to(device), y_train.to(device)

                # transformation
                x_train_mask = self.transform(x_train)

                # learning
                optimizer.zero_grad()
                loss_train = self.clf_model.get_loss(x_train_mask, y_train)
                loss_train.backward()
                optimizer.step()
                losses.append(loss_train)
                with torch.no_grad():
                    logits = self.clf_model(x_train_mask)
                    accuracy_train = (self.get_output(logits) == y_train).sum().item() / x_train.size()[0]
                    accuracies.append(accuracy_train)
                if verbose:
                    logger.info(
                        f'Mini batch: {i * nbatches + i_batch + 1}/{epochs * nbatches} | Training loss (batch level): {losses[-1]:.4f} | Train accuracy: {accuracy_train * 100:.2f}%.')

            self.clf_model.eval()
            avg_f1_val = []
            with torch.no_grad():
                for x_val, y_val in validation_x_y:
                    x_val, y_val = x_val.to(device), y_val.to(device)

                    y_votes = self.sample_funcs(x_val, n_sampling)
                    y_pred = y_votes.argmax(dim=-1)
                    # acc_val = (y_pred == y_val).sum().item() / float(x_val.size()[0])
                    f1_val = f1_score(y_val.cpu().numpy(), y_pred.cpu().numpy(), average='macro')
                    avg_f1_val.append(f1_val)
            avg_f1_val = np.mean(avg_f1_val)

            if avg_f1_val >= best_avg_f1:
                best_avg_f1 = avg_f1_val
                best_epoch = i
                torch.save(self.clf_model.state_dict(), self.model_save_path)
                logger.info(f'Model saved at path: {self.model_save_path}')

            logger.info(
                f'Validation accuracy: {avg_f1_val * 100:.2f}% | The best validation accuracy: {best_avg_f1 * 100:.2f}% at epoch: {best_epoch}.')

    def predict(self, x: torch.Tensor, n_sampling: int, alpha: float):
        self.eval()
        y_votes = self.sample_funcs(x, n_sampling)

        top2 = y_votes.argsort(dim=-1, descending=True)[:, :2]
        count1 = (y_votes[range(x.shape[0]), top2[:, 0]]).cpu().numpy()
        count2 = (y_votes[range(x.shape[0]), top2[:, 1]]).cpu().numpy()

        pred = top2[:, 0]
        prob_underlined = lower_confidence_interval(count1, n_sampling, alpha)
        prob_upperlined = upper_confidence_interval(count2, n_sampling, alpha)
        abstain_indicator = prob_underlined <= prob_upperlined
        # abstain_flag = binom_test(count1, count1 + count2, prop=0.5) > alpha
        pred[abstain_indicator] = RandomSmooth.ABSTAIN
        return pred

    def load_model(self):
        self.clf_model.load_state_dict(torch.load(self.model_save_path))

    def get_loss(self, x: torch.Tensor, label: int, n, batch_size=64):
        loss = 0.
        for i in range(n):
            mask_x = self.transform(x)
            loss += self.clf_model.get_loss(mask_x, label)
        return loss

    def get_confidence(self, x: torch.Tensor, n_sampling: int):
        y_votes = self.sample_funcs(x, n_sampling)
        return torch.softmax(y_votes.float(), dim=-1)

    def certify(self, x: torch.Tensor, labels: np.ndarray, n_selection: int, n_estimation: int,
                alpha: float,
                device='cpu'):
        assert n_selection > 0 and n_estimation > 0 and 0 < alpha < 1

        counts_selection = self.sample_funcs(x, n_selection)
        counts_estimation = self.sample_funcs(x, n_estimation)

        c_pred = counts_selection.argmax(dim=-1).cpu().numpy()
        n_targeted = counts_estimation[range(len(c_pred)), c_pred]
        prob_underlined = lower_confidence_interval(n_targeted.cpu().numpy(), n_estimation, alpha)
        n_runnerup = counts_estimation[range(len(c_pred)), 1 - c_pred]
        prob_overlined = upper_confidence_interval(n_runnerup.cpu().numpy(), n_estimation, alpha)

        # given the estimated probability, we calculate the radius
        radii = np.zeros_like(c_pred, dtype=object)
        abstain_indicator = prob_underlined <= prob_overlined
        radii[abstain_indicator] = HashSmooth.ABSTAIN

        incorrect_indicator = c_pred != labels.cpu().numpy()
        incorrect_indicator_true = incorrect_indicator == True
        if np.any(incorrect_indicator_true):
            incorrect_indicator[incorrect_indicator_true] = incorrect_indicator[incorrect_indicator_true] ^ \
                                                            abstain_indicator[incorrect_indicator_true]
            radii[incorrect_indicator] = self.ABSTAIN - 1
        total_abstain_indicator = abstain_indicator | incorrect_indicator
        if np.any(~total_abstain_indicator):
            radii[~total_abstain_indicator] = self.calc_radius(prob_underlined[~total_abstain_indicator],
                                                               prob_overlined[~total_abstain_indicator],
                                                               k_hashcode=self.k_hashcode,
                                                               max_radius=0.1,
                                                               n_grid=1000
                                                              )
        return radii, total_abstain_indicator

    def sample_funcs(self, x: torch.Tensor, n):
        votes = torch.zeros((x.shape[0], self.num_of_classes), dtype=torch.long, device=x.device)
        with torch.no_grad():
            if x.shape[0] > 1:
                for i in range(n):
                    x_train_mask = self.transform(x)
                    logits = self.clf_model(x_train_mask)
                    pred_batch = self.get_output(logits).to(torch.int64)
                    votes += torch.nn.functional.one_hot(pred_batch, self.num_of_classes)
            else:
                batch_size = 64
                for idx in range(n // batch_size + 1):
                    current_batch_size = min(batch_size, n)
                    n -= current_batch_size
                    if current_batch_size <= 0:
                        break
                    mask_x = self.transform(torch.tile(x, (current_batch_size, 1)))
                    logits = self.clf_model(mask_x)
                    pred_batch = self.get_output(logits).to(torch.int64)
                    votes += torch.sum(torch.nn.functional.one_hot(pred_batch, self.num_of_classes), dim=0,
                                       keepdim=True)
        return votes

    def get_output(self, logits: torch.Tensor):
        if isinstance(self.clf_model, DrebinSVM):
            return (logits.data > 0.).view(-1)
        elif isinstance(self.clf_model, DrebinNN):
            return logits.argmax(dim=-1)
        else:
            raise TypeError("Expect 'DrebinSVM' or 'DrebinNN'.")