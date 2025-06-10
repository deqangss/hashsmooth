import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np


def get_simple_fc_model(input_dim=128, output_dim=10000):
    model = nn.Sequential(
        nn.Linear(input_dim, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, output_dim),
        nn.Sigmoid()
    )
    return model


class SimpleDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, item):
        return self.x[item], self.y[item]

    def __len__(self):
        return len(self.x)


def train_sample_model(model: nn.Module, x: np.ndarray, y: np.ndarray,
                       batch_size=64,
                       epoch=10000,
                       learning_rate=0.001):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x_tensor = torch.from_numpy(x).float().to(device)
    y_tensor = torch.from_numpy(y).float().to(device)
    dataloader = DataLoader(SimpleDataset(x_tensor, y_tensor), batch_size=64, shuffle=True)
    model = model.to(device)
    model.train()
    loss = nn.MSELoss()
    optimzier = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for idx in range(epoch):
        losses_list = []
        for data in dataloader:
            x_tensor_batch, y_tensor_batch = data
            model.train().zero_grad()
            error = loss(model(x_tensor_batch), y_tensor_batch)
            error.backward()
            optimzier.step()
            losses_list.append(error.sum().item())
        loss_mean = np.mean(losses_list)
        print(f"Epoch {idx} with the loss {loss_mean:.4f}.")
    return
