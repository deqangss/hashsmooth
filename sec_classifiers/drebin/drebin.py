import os
import math
import numpy as np
import torch
from statsmodels.stats.proportion import proportion_confint, binom_test

from hashsmooth.classifier_template import BasicClassifier
from hashsmooth.core import HashSmooth
from sec_classifiers.hrat_malscan.randomsmooth_hrat_malscan_det_fs import RandomTransformer, RandomSmooth
from tools import utils
from hashsmooth.utils_hash import lower_confidence_interval, upper_confidence_interval

binom_test = np.vectorize(binom_test)


class DrebinSVM(BasicClassifier):
    def __init__(self, input_dim: int, num_classes=2, batch_size=64, model_save_dir=''):
        BasicClassifier.__init__(self, batch_size)
        self.input_dim = input_dim
        assert num_classes == 1
        self.num_classes = num_classes
        self.model_save_dir = model_save_dir
        utils.mkdir(self.model_save_dir)
        self.model_save_path = os.path.join(self.model_save_dir, 'model.ckpt')

        self.model = torch.nn.Linear(self.input_dim, 1)
        self.criterion = torch.nn.BCEWithLogitsLoss()  # better than hinge loss

    def fit(self, train_x_y: torch.utils.data.dataloader,
            validation_x_y: torch.utils.data.dataloader,
            epochs=1000, learning_rate=0.001, penaty_factor=0.01, device='cpu', verbose=False):
        nbatches = len(train_x_y)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
        best_avg_acc = 0.
        for i in range(epochs):
            self.model.train()
            losses, accuraies = [], []
            for i_batch, (x_train, y_train) in enumerate(train_x_y):
                x_train, y_train = x_train.to(device), y_train.to(device)
                optimizer.zero_grad()
                logits = self.model(x_train)
                loss_train = self.criterion(logits.view(-1), y_train.float())
                weight = self.model.weight.squeeze()
                loss_train += penaty_factor * torch.sum(weight * weight)
                loss_train.backward()
                optimizer.step()
                accuray_train = ((logits.data > 0.5).view(-1) == y_train).sum().item() / x_train.size()[0]
                accuraies.append(accuray_train)
                losses.append(loss_train)

                if verbose:
                    print(
                        f'Mini batch: {i * nbatches + i_batch + 1}/{epochs * nbatches} | Training loss (batch level): {losses[-1]:.4f} | Train accuracy: {accuray_train * 100:.2f}')

            self.model.eval()
            avg_acc_val = []
            with torch.no_grad():
                for x_val, y_val in validation_x_y:
                    x_val, y_val = x_val.to(device), y_val.to(device)
                    logits = self.model(x_val)
                    acc_val = ((logits.data > 0.5).view(-1) == y_val).sum().item()
                    acc_val /= x_val.size()[0]
                    avg_acc_val.append(acc_val)
                avg_acc_val = np.mean(avg_acc_val)

                if avg_acc_val >= best_avg_acc:
                    best_avg_acc = avg_acc_val
                    best_epoch = i
                    torch.save(self.model.state_dict(), self.model_save_path)
                    print(f'Model saved at path: {self.model_save_path}')

                if verbose:
                    print(
                        f'Validation accuracy: {avg_acc_val * 100:.2f} | The best validation accuracy: {best_avg_acc * 100:.2f} at epoch: {best_epoch}')

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(x)
        x_preds = (logits.data > 0.5).view(-1).to(torch.int)
        return x_preds

    def load_model(self):
        self.model.load_state_dict(torch.load(self.model_save_path))

    def get_loss(self, x:torch.Tensor, label: int):
        logits = self.model(x)
        return self.criterion(logits.view(-1), label.float())

    def certify(self, x: np.ndarray, label: (int, np.ndarray)):
        raise NotImplementedError

    def eval(self):
        self.model.eval()


class DrebinNN(BasicClassifier):
    def __init__(self, input_dim: int, num_classes=2, batch_size=64, model_save_dir=''):
        BasicClassifier.__init__(self, batch_size)
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.model_save_dir = model_save_dir
        utils.mkdir(self.model_save_dir)
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

    def fit(self, train_x_y: torch.utils.data.dataloader,
            validation_x_y: torch.utils.data.dataloader,
            epochs=1000, learning_rate=0.001, device='cpu', verbose=False):
        nbatches = len(train_x_y)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        best_avg_acc = 0.
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
                print(logits.argmax(dim=-1), y_train)
                exit(-1)
                accuray_train = (logits.argmax(dim=-1) == y_train).sum().item() / x_train.shape[0]
                accuracies.append(accuray_train)
                losses.append(loss_train)

                if verbose:
                    print(
                        f'Mini batch: {i * nbatches + i_batch + 1}/{epochs * nbatches} | Training loss (batch level): {losses[-1]:.4f} | Train accuracy: {accuray_train * 100:.2f}')

            self.model.eval()
            avg_acc_val = []
            with torch.no_grad():
                for x_val, y_val in validation_x_y:
                    x_val, y_val = x_val.to(device), y_val.to(device)

                    logits = self.model(x_val)
                    acc_val = (logits.argmax(dim=-1) == y_val).sum().item()
                    print(logits.argmax(dim=-1), y_val, x_val.size()[0])
                    exit(-1)
                    acc_val /= x_val.size()[0]
                    avg_acc_val.append(acc_val)
                avg_acc_val = np.mean(avg_acc_val)

                if avg_acc_val >= best_avg_acc:
                    best_avg_acc = avg_acc_val
                    best_epoch = i
                    torch.save(self.model.state_dict(), self.model_save_path)
                    print(f'Model saved at path: {self.model_save_path}')

                if verbose:
                    print(
                        f'Validation accuracy: {avg_acc_val * 100:.2f} | The best validation accuracy: {best_avg_acc * 100:.2f} at epoch: {best_epoch}')

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(x)
        x_preds = logits.argmax(dim=-1).to(torch.int)
        return x_preds

    def load_model(self):
        self.model.load_state_dict(torch.load(self.model_save_path))

    def get_loss(self, x: torch.Tensor, label: int):
        logits = self.model(x)
        return self.criterion(logits, label.to(torch.long))

    def certify(self, x: np.ndarray, label: (int, np.ndarray)):
        raise NotImplementedError

    def eval(self):
        self.model.eval()


class RandomSmooth4Drebin(RandomSmooth):
    def __init__(self, drebin_clf_model, num_of_classes: int, transform_method: RandomTransformer, max_k: int,
                 default_mode: bool, model_save_dir=''):
        """
        Customized randomsmooth w.r.t. drebin
        :param drebin_clf_model: classification model
        :param num_of_classes: number of classes
        :param transform_method: input transformation model
        :param max_k: number of maximum elements
        :param default_mode: miss-classification threshold is 0.5 or not
        """
        RandomSmooth.__init__(self, drebin_clf_model, num_of_classes, transform_method, max_k, default_mode)
        self.model_save_dir = model_save_dir
        utils.mkdir(self.model_save_dir)
        self.model_save_path = os.path.join(self.model_save_dir, 'model.ckpt')

    def eval(self):
        self.base_classifier.model.eval()

    def get_output(self, logits: torch.Tensor):
        if isinstance(self.base_classifier, DrebinSVM):
            return (logits.data > 0.5).view(-1)
        elif isinstance(self.base_classifier, DrebinNN):
            return logits.argmax(dim=-1)
        else:
            raise TypeError("Expect 'DrebinSVM' or 'DrebinNN'.")

    def fit(self, train_x_y: torch.utils.data.dataloader,
            validation_x_y: torch.utils.data.dataloader,
            epochs=100, learning_rate=0.001, n_sampling=100, device='cpu', verbose=False):
        nbatches = len(train_x_y)
        if isinstance(self.base_classifier, DrebinSVM):
            optimizer = torch.optim.SGD(self.base_classifier.model.parameters(), lr=learning_rate)
        elif isinstance(self.base_classifier, DrebinNN):
            optimizer = torch.optim.Adam(self.base_classifier.model.parameters(), lr=learning_rate)
        else:
            raise TypeError("Expect 'DrebinSVM' or 'DrebinNN'.")
        best_avg_acc = 0.
        for i in range(epochs):
            self.base_classifier.model.train()
            losses, accuracies = [], []
            for i_batch, (x_train, y_train) in enumerate(train_x_y):
                x_train, y_train = x_train.to(device), y_train.to(device)
                x_train_mask = self.transform_method.transform(x_train)

                optimizer.zero_grad()
                loss_train = self.base_classifier.get_loss(x_train_mask, y_train)
                loss_train.backward()
                optimizer.step()
                losses.append(loss_train)
                with torch.no_grad():
                    logits = self.base_classifier.model(x_train_mask)
                    accuracy_train = (self.get_output(logits) == y_train).sum().item() / x_train.size()[0]
                    accuracies.append(accuracy_train)
                if verbose:
                    print(
                        f'Mini batch: {i * nbatches + i_batch + 1}/{epochs * nbatches} | Training loss (batch level): {losses[-1]:.4f} | Train accuracy: {accuracy_train * 100:.2f}')

            self.base_classifier.model.eval()
            avg_acc_val = []
            with torch.no_grad():
                for x_val, y_val in validation_x_y:
                    x_val, y_val = x_val.to(device), y_val.to(device)
                    y_votes, _1 = self.sample_funcs(x_val, n_sampling)
                    y_pred = y_votes.argmax(dim=-1)
                    acc_val = (y_pred == y_val).sum().item() / float(x_val.size()[0])
                    acc_val /= x_val.size()[0]
                    avg_acc_val.append(acc_val)
            avg_acc_val = np.mean(avg_acc_val)

            if avg_acc_val >= best_avg_acc:
                best_avg_acc = avg_acc_val
                best_epoch = i
                torch.save(self.base_classifier.model.state_dict(), self.model_save_path)
                print(f'Model saved at path: {self.model_save_path}')

            if verbose:
                print(
                    f'Validation accuracy: {avg_acc_val * 100:.2f} | The best validation accuracy: {best_avg_acc * 100:.2f} at epoch: {best_epoch}')

    def predict(self, x: np.ndarray, n_sampling: int, alpha: float):
        y_votes, _2 = self.sample_funcs(x, n_sampling)

        top2 = y_votes.argsort(dim=-1, descending=True)[:, :2]
        count1 = (y_votes[range(x.shape[0]), top2[:, 0]]).cpu().numpy()
        count2 = (y_votes[range(x.shape[0]), top2[:, 1]]).cpu().numpy()

        pred = top2[:, 0]
        abstain_flag = binom_test(count1, count1 + count2, prop=0.5) > alpha
        pred[abstain_flag] = RandomSmooth.ABSTAIN
        return pred

    def load_model(self):
        self.base_classifier.model.load_state_dict(torch.load(self.model_save_path))

    def get_loss(self, x: torch.Tensor, label: int, n, k_per_instance=0, batch_size=64):
        if 0 < k_per_instance <= 1:
            k_per_instance = min(math.ceil(len(x) * k_per_instance), self.max_k)
        loss = 0.
        for i in range(n):
            mask_x = self.transform_method.transform(x, k_per_instance)
            loss += self.base_classifier.get_loss(mask_x, label)
        return loss

    def certify(self, x: np.ndarray, labels: np.ndarray, n_selection: int, n_estimation: int,
                k_per_instance: (float, int),
                alpha: float,
                device='cpu'):
        if 0 < k_per_instance <= 1:
            k_per_instance = min(math.ceil(len(x) * k_per_instance), self.max_k)
        with torch.no_grad():
            counts_selection, _1 = self.sample_funcs(x, n_selection)
            counts_estimation, _2 = self.sample_funcs(x, n_estimation)

            c_pred = counts_selection.argmax(dim=-1)
            n_targeted = counts_estimation[range(len(c_pred)), c_pred]

            prob_underlined = lower_confidence_interval(n_targeted.cpu().numpy(), n_estimation, alpha)

            radius = self.population_radius_for_majority(np.array(prob_underlined)[..., None],
                                                         n_estimation,
                                                         k_per_instance)

            radius[(c_pred != labels).cpu().numpy()] = RandomSmooth.ABSTAIN
            return radius

    def sample_funcs(self, x: torch.Tensor, n, k_per_instance=0):
        assert n > 0
        if 0 < k_per_instance <= 1:
            k_per_instance = min(math.ceil(len(x) * k_per_instance), self.max_k)
        votes = torch.zeros((x.shape[0], self.num_of_classes), dtype=torch.long, device=x.device)
        with torch.no_grad():
            for i in range(n):
                mask_x = self.transform_method.transform(x, k_per_instance)
                logits = self.base_classifier.model(mask_x)
                pred_batch = self.get_output(logits).to(torch.int64)
                votes += torch.nn.functional.one_hot(pred_batch, self.num_of_classes)
        return votes, k_per_instance


class HashSmooth4Drebin(HashSmooth):
    def __init__(self,
                 drebin_clf_model,
                 num_of_classes: int,
                 hash_methods: list,
                 n_subfeatures: list,
                 k_hashcode: int,
                 max_k,
                 max_radii: list,
                 n_grids: list,
                 default_mode: bool,
                 model_save_dir=''
                 ):
        """
        Customized HashSmooth w.r.t. Drebin
        """
        HashSmooth.__init__(self,
                            drebin_clf_model,
                            num_of_classes,
                            hash_methods,
                            n_subfeatures=n_subfeatures,
                            k_hashcode=k_hashcode,
                            max_k=max_k,
                            max_radii=max_radii,
                            n_grids=n_grids,
                            default_mode=default_mode)

        self.model_save_dir = model_save_dir
        utils.mkdir(self.model_save_dir)
        self.model_save_path = os.path.join(self.model_save_dir, 'model.ckpt')

    def eval(self):
        self.base_classifier.model.eval()

    def fit(self, train_x_y: torch.utils.data.dataloader, validation_x_y: torch.utils.data.dataloader,
            epochs=1000, learning_rate=0.001, n_sampling=100, device='cpu', verbose=False):
        nbatches = len(train_x_y)
        if isinstance(self.base_classifier, DrebinSVM):
            optimizer = torch.optim.SGD(self.base_classifier.model.parameters(), lr=learning_rate)
        elif isinstance(self.base_classifier, DrebinNN):
            optimizer = torch.optim.Adam(self.base_classifier.model.parameters(), lr=learning_rate)
        else:
            raise TypeError("Expect 'DrebinSVM' or 'DrebinNN'.")
        best_avg_acc = 0.
        for i in range(epochs):
            self.base_classifier.model.train()
            losses, accuracies = [], []
            for i_batch, (x_train, y_train) in enumerate(train_x_y):
                x_train, y_train = x_train.to(device), y_train.to(device)

                # transformation
                x_train_mask = self.transform_wrapper(x_train)

                # learning
                optimizer.zero_grad()
                loss_train = self.base_classifier.get_loss(x_train_mask, y_train)
                loss_train.backward()
                optimizer.step()
                losses.append(loss_train)
                with torch.no_grad():
                    logits = self.base_classifier.model(x_train_mask)
                    accuracy_train = (self.get_output(logits) == y_train).sum().item() / x_train.size()[0]
                    accuracies.append(accuracy_train)
                if verbose:
                    print(
                        f'Mini batch: {i * nbatches + i_batch + 1}/{epochs * nbatches} | Training loss (batch level): {losses[-1]:.4f} | Train accuracy: {accuracy_train * 100:.2f}')

            self.base_classifier.model.eval()
            avg_acc_val = []
            with torch.no_grad():
                for x_val, y_val in validation_x_y:
                    x_val, y_val = x_val.to(device), y_val.to(device)

                    y_votes = self.sample_funcs(x_val,  n_sampling)
                    y_pred = y_votes.argmax(dim=-1)
                    acc_val = (y_pred == y_val).sum().item() / float(x_val.size()[0])
                    avg_acc_val.append(acc_val)
            avg_acc_val = np.mean(avg_acc_val)

            if avg_acc_val >= best_avg_acc:
                best_avg_acc = avg_acc_val
                best_epoch = i
                torch.save(self.base_classifier.model.state_dict(), self.model_save_path)
                print(f'Model saved at path: {self.model_save_path}')

            if verbose:
                print(
                    f'Validation accuracy: {avg_acc_val * 100:.2f} | The best validation accuracy: {best_avg_acc * 100:.2f} at epoch: {best_epoch}')

    def predict(self, x: np.ndarray, n_sampling: int, alpha: float):
        y_votes = self.sample_funcs(x, n_sampling)

        top2 = y_votes.argsort(dim=-1, descending=True)[:, :2]
        count1 = (y_votes[range(x.shape[0]), top2[:, 0]]).cpu().numpy()
        count2 = (y_votes[range(x.shape[0]), top2[:, 1]]).cpu().numpy()

        pred = top2[:, 0]
        abstain_flag = binom_test(count1, count1 + count2, prop=0.5) > alpha
        pred[abstain_flag] = RandomSmooth.ABSTAIN
        return pred

    def load_model(self):
        self.base_classifier.model.load_state_dict(torch.load(self.model_save_path))

    def get_loss(self, x: torch.Tensor, label: int, n, batch_size=64):
        loss = 0.
        for i in range(n):
            mask_x = self.transform_wrapper(x)
            loss += self.base_classifier.get_loss(mask_x, label)
        return loss

    def certify(self, x: np.ndarray, labels: np.ndarray, n_selection: int, n_estimation: int,
                k_per_instance: (float, int),
                alpha: float,
                device='cpu'):
        assert n_selection > 0 and n_estimation > 0 and 0 < alpha < 1

        counts_selection = self.sample_funcs(x, n_selection)
        counts_estimation = self.sample_funcs(x, n_estimation)

        c_pred = counts_selection.argmax(dim=-1).cpu().numpy()
        n_targeted = counts_estimation[range(len(c_pred)), c_pred]

        # given the estimated probability, we calculate the radius
        prob_underlined = lower_confidence_interval(n_targeted.cpu().numpy(), n_estimation, alpha)

        if self.default_mode:
            abstain_indicator = prob_underlined <= 0.5
            c_pred[abstain_indicator] = HashSmooth.ABSTAIN
            radii = np.ones_like(c_pred, dtype=object)
            radii[abstain_indicator] = 0.
            radii[~abstain_indicator] = self._calc_radius(prob_underlined[~abstain_indicator], 0.5,
                                                          [],
                                                          [],
                                                          []
                                                          )
        else:
            c_pred_runnerup = counts_selection.argsort()[:, -2]
            n_targeted_runnerup = counts_estimation[c_pred_runnerup]
            prob_upperlined = upper_confidence_interval(n_targeted_runnerup, c_pred_runnerup, alpha)
            abstain_indicator = prob_underlined <= prob_upperlined
            c_pred[abstain_indicator] = HashSmooth.ABSTAIN
            radii = np.ones_like(c_pred, dtype=object)
            radii[abstain_indicator] = 0.
            radii[abstain_indicator] = self._calc_radius(prob_underlined[abstain_indicator], prob_upperlined,
                                                         [],
                                                         [],
                                                         []
                                                         )
        return radii

    def sample_funcs(self, x: torch.Tensor, n):
        votes = torch.zeros((x.shape[0], self.num_of_classes), dtype=torch.long, device=x.device)
        with torch.no_grad():
            for i in range(n):
                x_train_mask = self.transform_wrapper(x)
                logits = self.base_classifier.model(x_train_mask)
                pred_batch = self.get_output(logits).to(torch.int64)
                votes += torch.nn.functional.one_hot(pred_batch, self.num_of_classes)
        return votes

    def get_output(self, logits: torch.Tensor):
        if isinstance(self.base_classifier, DrebinSVM):
            return (logits.data > 0.5).view(-1)
        elif isinstance(self.base_classifier, DrebinNN):
            return logits.argmax(dim=-1)
        else:
            raise TypeError("Expect 'DrebinSVM' or 'DrebinNN'.")
