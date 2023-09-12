import os
import random
import tempfile

import numpy as np
import torch

from tools.utils import read_pickle, dump_pickle, guess_file_name


class DatasetTorch(torch.utils.data.Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        x = self.data[item]
        y = self.label[item]
        return x, y


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, dataset_name='malscan', batch_size=64):
        """
        dataset configurator
        :param dataset_dir: dataset folder
        :param dataset_name: dataset name
        """
        self.dataset_dir = dataset_dir
        self.dataset_name = dataset_name
        self.dataset_path = os.path.join(self.dataset_dir, self.dataset_name)
        assert os.path.exists(self.dataset_path)
        self.batch_size = batch_size

    def load(self):
        # Here we just load the preprocessed data, while the data preprocessing shall be conducted by following the
        # previous studies
        train_data_path = os.path.join(self.dataset_path, 'train.pkl')
        dir_file_names = os.path.dirname(self.dataset_path)
        if not os.path.exists(train_data_path):
            # guess it
            g_file_names = guess_file_name(self.dataset_path, "train")
            raise FileExistsError("Only load 'train.pkl', and do you mean load {}.\n".format(','.join(g_file_names)))

        train_dataset = read_pickle(train_data_path)

        val_data_path = os.path.join(self.dataset_path, 'validation.pkl')
        if not os.path.exists(val_data_path):
            # guess it
            g_file_names = guess_file_name(self.dataset_path, "validation")
            raise FileExistsError("Only load 'validation.pkl', and do you mean load {}.\n".format(','.join(g_file_names)))
        val_dataset = read_pickle(val_data_path)

        test_data_path = os.path.join(self.dataset_path, 'test.pkl')
        if not os.path.exists(val_data_path):
            # guess it
            g_file_names = guess_file_name(self.dataset_path, "test")
            raise FileExistsError(
                "Only load 'test.pkl', and do you mean load {}.\n".format(','.join(g_file_names)))
        test_dataset = read_pickle(test_data_path)
        return train_dataset, val_dataset, test_dataset

    def get_dataloader(self, data, label=None):
        data_params = {
            'batch_size': self.batch_size,
            'shuffle': False,
            'num_workers': 6
        }
        assert isinstance(data, (np.ndarray, torch.Tensor))
        if label is None:
            return torch.utils.data.DataLoader(data, **data_params)
        elif isinstance(label, (list, np.ndarray, torch.Tensor)):
            return torch.utils.data.DataLoader(
                DatasetTorch(data, label), **data_params)
        else:
            raise ValueError


