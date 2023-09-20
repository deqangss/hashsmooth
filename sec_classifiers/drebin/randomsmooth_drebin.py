import time
import numpy as np
import torch
import warnings

from sec_classifiers.drebin.drebin import DrebinSVM
from tools import utils


class RandomDrebinSVM(DrebinSVM):
    def __init__(self, input_dim: int, num_classes=1, batch_size=64, model_save_path=''):
        BasicClassifier.__init__(batch_size)
        torch.nn.Module.__init__()
        assert num_classes == 1, "Expected binary classification.\n"
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.model_save_path = model_save_path
        utils.mkdir(self.model_save_path)

        self.linear = torch.nn.Linear(self.input_dim, 1)
        self.criterion = torch.nn.HingeEmbeddingLoss()

    def fit(self, train_x_y: torch.utils.data.dataloader,
              validation_x_y: torch.utils.data.dataloader,
              epochs=1000, learning_rate=0.001, device='cpu', verbose=False):
        nbatches = len(train_x_y)
        optimizer = torch.optim.SGD(self.parameters(), lr = learning_rate)
        for i in range(epochs):
            self.train()
            losses, accuraies = [], []
            for i_batch, (x_train, y_train) in enumerate(self.train_x_y):
                x_train, y_train = x_train.to(device), y_train.to(device)
                optimizer.zero_grad()
                logits = self.linear(x_train)
                loss_train = self.criterion(logits, y_train)
                loss_train.backward()
                optimizer.step()

                accuray_train = ((logits.data > 0).view(-1) == y_train).sum().item() / x_train.size()[0]
                accuraies.append(accuray_train)
                losses.append(loss_train)

                if verbose:
                    print(f'Mini batch: {i * nbatches + i_batch + 1}/{epochs * nbatches} | Training loss (batch level): {losses[-1]:.4f} | Train accuracy: {accuray_train * 100:.2f}')

            self.eval()
            avg_acc_val = []
            with torch.no_grad():
                for x_val, y_val in validation_x_y:
                    x_val, y_val = x_val.to(device), y_val.to(device)
                    logits = self.linear(x_val)
                    acc_val = ((logits.data > 0).view(-1) == y_val).sum().item()
                    acc_val /= x_val.size()[0]
                    avg_acc_val.append(acc_val)
                avg_acc_val = np.mean(avg_acc_val)

                if avg_acc_val >= best_avg_acc:
                    best_avg_acc = avg_acc_val
                    best_epoch = i
                    torch.save(self.state_dict(), self.model_save_path)
                    if verbose:
                        print(f'Model saved at path: {self.model_save_path}')

                if verbose:
                    print(f'Validation accuracy: {avg_acc_val * 100:.2f} | The best validation accuracy: {best_avg_acc * 100:.2f} at epoch: {best_epoch}')

    def predict(self, x: torch.Tensor, device='cpu') -> torch.Tensor:
        x = x.to(device)
        self.load_state_dict(torch.load(self.model_save_path))
        logits = self.linear(x)
        x_preds = (logits.data > 0).view(-1).to(torch.int)
        return x_preds

    def certify(self, x: np.ndarray, label: (int, np.ndarray)):
        raise NotImplementedError

    def eval(self):
        raise NotImplementedError