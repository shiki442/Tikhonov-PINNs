import os

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset, Dataset, IterableDataset
from torch.utils.data import DistributedSampler


class TikDataset(Dataset):
    """Dataset that stores interior and boundary points separately.

    This allows independent sampling of interior and boundary points
    without forcing them to have the same number of samples.
    """
    def __init__(self, int_data: Tensor, bdy_data: Tensor):
        """
        Args:
            int_data: Interior data tensor of shape (n_int, cols_int)
            bdy_data: Boundary data tensor of shape (n_bdy, cols_bdy)
        """
        self.int_data = int_data
        self.bdy_data = bdy_data

    def __len__(self):
        return len(self.int_data)

    def __getitem__(self, idx):
        # Return paired samples - interior at idx, boundary at modulated idx
        int_idx = idx % len(self.int_data)
        bdy_idx = idx % len(self.bdy_data)
        return self.int_data[int_idx], self.bdy_data[bdy_idx]


def tik_collate_fn(batch):
    """
    Collate function for TikDataset.

    Args:
        batch: List of tuples [(int_sample, bdy_sample), ...]

    Returns:
        Tuple of (int_batch, bdy_batch) where each is a stacked tensor
    """
    int_samples, bdy_samples = zip(*batch)
    return torch.stack(int_samples), torch.stack(bdy_samples)


def get_dataloader(data_path, batch_size, idx, noise_str, n_samples):
    """
    Load data and create DataLoader.
    Supports both .pt (dictionary format) and .txt (legacy matrix format) files.
    """
    pt_path = os.path.join(data_path, f"example{idx}_data{noise_str}.pt")
    txt_path = os.path.join(data_path, f"example{idx}data{noise_str}.txt")

    # Prefer .pt format if exists
    if os.path.exists(pt_path):
        dataset = _load_dataset_pt(pt_path, n_samples)
    elif os.path.exists(txt_path):
        dataset = _load_dataset_txt(txt_path, n_samples)
    else:
        raise FileNotFoundError(f"Data file not found: {pt_path} or {txt_path}")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=tik_collate_fn)
    return dataloader


def get_ddp_dataloader(data_path, batch_size, idx, noise_str, n_samples, world_size, rank):
    """
    Load data and create DataLoader with DistributedSampler for DDP training.
    Supports both .pt and .txt formats.
    """
    pt_path = os.path.join(data_path, f"example{idx}_data{noise_str}.pt")
    txt_path = os.path.join(data_path, f"example{idx}data{noise_str}.txt")

    # Prefer .pt format if exists
    if os.path.exists(pt_path):
        dataset = _load_dataset_pt(pt_path, n_samples)
    elif os.path.exists(txt_path):
        dataset = _load_dataset_txt(txt_path, n_samples)
    else:
        raise FileNotFoundError(f"Data file not found: {pt_path} or {txt_path}")

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, sampler=sampler, num_workers=0, collate_fn=tik_collate_fn)
    return dataloader


def _load_dataset_pt(file_path, n_samples=-1):
    """
    Load dataset from .pt file (dictionary format).

    Supports both 1D and n-dimensional data.

    Returns a TikDataset that stores interior and boundary data separately,
    allowing independent sampling without forcing equal sample counts.

    Interior data format (each row):
        [int_x1, ..., int_xd, m_int, f_val, u_dagger, q_dagger]
        Total: d + 4 columns

    Boundary data format (each row):
        [bdy_x1, ..., bdy_xd, normal_x1, ..., normal_xd, m_bdy, g_val]
        Total: 2*d + 2 columns
    """
    data_dict = torch.load(file_path, weights_only=False)

    # Extract data from dictionary
    int_points = data_dict['int_points']  # (n_int, dim)
    bdy_points = data_dict['bdy_points']  # (n_bdy, dim)
    normal_vec = data_dict['normal_vec']  # (n_bdy, dim)
    m_int = data_dict['m_int']            # (n_int, 1)
    m_bdy = data_dict['m_bdy']            # (n_bdy, 1)
    f_val = data_dict['f_val']            # (n_int, 1)
    g_val = data_dict['g_val']            # (n_bdy, 1)
    u_dagger = data_dict['u_dagger']      # (n_int, 1)
    q_dagger = data_dict['q_dagger']      # (n_int, 1)

    n_int = int_points.shape[0]
    n_bdy = bdy_points.shape[0]

    # Apply sample limit if specified (before shuffling)
    if n_samples > 0:
        if n_samples < n_int:
            # Subsample interior data
            rand_index_int = np.random.permutation(n_int)[:n_samples]
            int_points = int_points[rand_index_int]
            m_int = m_int[rand_index_int]
            f_val = f_val[rand_index_int]
            u_dagger = u_dagger[rand_index_int]
            q_dagger = q_dagger[rand_index_int]
            n_int = n_samples
        # Note: boundary data keeps its original size, no subsampling

    # Build interior data tensor: [int_x, m_int, f_val, u_dagger, q_dagger]
    int_data = torch.cat([int_points, m_int, f_val, u_dagger, q_dagger], dim=1)

    # Build boundary data tensor: [bdy_x, normal, m_bdy, g_val]
    bdy_data = torch.cat([bdy_points, normal_vec, m_bdy, g_val], dim=1)

    return TikDataset(int_data, bdy_data)


def _load_dataset_txt(file_path, n_samples=-1):
    """Load dataset from legacy .txt file (matrix format)."""
    data = np.loadtxt(file_path, dtype='float', delimiter=',')
    if n_samples > 0 and n_samples < data.shape[0]:
        data = data[:n_samples, :]
    return TensorDataset(Tensor(data))
