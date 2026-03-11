import os

import numpy as np
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import DataLoader, DistributedSampler, Dataset


def get_dataloader(data_path, batch_size, idx, noise_str, n_samples):
    data_path = os.path.join(data_path, "example" + idx + "data" + noise_str + ".txt")
    dataset = _load_dataset(data_path, n_samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader


def get_ddp_dataloader(data_path, batch_size, idx, noise_str, n_samples, world_size, rank):
    data_path = os.path.join(data_path, "example" + idx + "data" + noise_str + ".txt")
    dataset = _load_dataset(data_path, n_samples)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=0)
    return dataloader


def _load_dataset(file_path, n_samples=-1):
    data = np.loadtxt(file_path, dtype='float', delimiter=',')
    if n_samples > 0 and n_samples < data.shape[0]:
        data = data[:n_samples, :]
    return TensorDataset(Tensor(data))
