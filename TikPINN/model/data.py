import os

import numpy as np
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import DataLoader, DistributedSampler, Dataset


def get_dataloader(data_path, batch_size, idx, noise_str):
    data_path = os.path.join(data_path, "example" + idx + "data" + noise_str + ".txt")
    dataset = _load_dataset(data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader


def get_ddp_dataloader(data_path, batch_size, idx, noise_str, world_size, rank):
    data_path = os.path.join(data_path, "example" + idx + "data" + noise_str + ".txt")
    dataset = _load_dataset(data_path)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=0)
    return dataloader


def _load_dataset(file_path):
    data = np.loadtxt(file_path, dtype='float', delimiter=',')
    return TensorDataset(Tensor(data))
