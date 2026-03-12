import os
import time

import numpy as np
import torch

from .loss import relative_error_u, relative_error_q
from .problem import evaluate_u, evaluate_q
from .utils import u_dagger, q_dagger, relative_error
from torch.optim.lr_scheduler import LambdaLR

# from torch.utils.tensorboard import SummaryWriter


@torch.no_grad()
def _generate_uniform_mesh(n: int, d: int = 2):
    if d == 1:
        grid_tmp = torch.linspace(0.0, 1.0, n + 1)
        return grid_tmp.reshape((-1, 1))
    elif d == 2:
        grid_tmp = torch.linspace(0.0, 1.0, n + 1)
        grid_x1, grid_x2 = torch.meshgrid([grid_tmp, grid_tmp], indexing='ij')
        grid_x1, grid_x2 = grid_x1.reshape((-1, 1)), grid_x2.reshape((-1, 1))
        return torch.cat((grid_x1, grid_x2), dim=1)
    else:
        raise ValueError(f"Unsupported dimension: {d}")


def _save_results(device, epoch, q_net, u_net, results_path):
    d = q_net.in_layer.in_features
    x = _generate_uniform_mesh(500, d).to(device)
    q_val = evaluate_q(q_net, x)
    u_val = evaluate_u(u_net, x)
    results = torch.cat((x, q_val, u_val), dim=1)
    results = results.detach().cpu().numpy()
    print(results_path)
    np.savetxt(os.path.join(results_path, f"result{epoch:>05d}.txt"), results, delimiter=",")


def _pretrain_u_epoch(device, dataloader, u_net, loss_fn, optimizer) -> None:
    """Train one epoch"""
    u_net.train()
    for _, samples in enumerate(dataloader):
        # load data to CUDA - samples is now (int_sample, bdy_sample) tuple
        samples = (samples[0].to(device), samples[1].to(device))
        # compute loss
        loss = loss_fn.measurement(u_net, samples)
        # backward and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def _pretrain_validate_u_epoch(device, dataloader, u_net, loss_fn, idx) -> tuple:
    """Validate"""
    u_net.eval()
    dataset_sizes = 0
    batch_loss = 0.0
    d = u_net.in_layer.in_features
    x = _generate_uniform_mesh(500, d).to(device)
    err_u = relative_error(u_net(x), u_dagger(x, idx))

    for _, samples in enumerate(dataloader):
        # load data to CUDA - samples is now (int_sample, bdy_sample) tuple
        samples = (samples[0].to(device), samples[1].to(device))
        # forward and compute loss
        loss = loss_fn.measurement(u_net, samples)
        # compute step loss and accuracy
        dataset_sizes += samples[0].size(0)
        batch_loss += loss.item() * samples[0].size(0)
    # compute epoch loss and relative error
    epoch_loss = batch_loss / dataset_sizes
    return (epoch_loss, err_u.item())


def _train_epoch(device, dataloader, q_net, u_net, loss_fn, optimizer, scheduler) -> None:
    """Train one epoch"""
    q_net.train()
    u_net.train()
    for _, samples in enumerate(dataloader):
        # samples is now (int_sample, bdy_sample) tuple

        def closure():
            # compute loss
            loss = loss_fn(q_net, u_net, samples)
            # backward and optimization
            optimizer.zero_grad()
            loss.backward()
            return loss

        # load data to CUDA - samples is now (int_sample, bdy_sample) tuple
        samples = (samples[0].to(device), samples[1].to(device))
        optimizer.step(closure)

    if scheduler is not None:
        scheduler.step()


def _validate_epoch(device, dataloader, q_net, u_net, loss_fn, idx) -> tuple:
    """Validate"""
    q_net.eval()
    u_net.eval()
    dataset_sizes = 0
    batch_loss = 0.0
    d = u_net.in_layer.in_features
    x = _generate_uniform_mesh(500, d).to(device)
    err_u = relative_error(u_net(x), u_dagger(x, idx))
    err_q = relative_error(q_net(x), q_dagger(x, idx))

    for _, samples in enumerate(dataloader):
        # load data to CUDA - samples is now (int_sample, bdy_sample) tuple
        samples = (samples[0].to(device), samples[1].to(device))
        # forward and compute loss
        loss = loss_fn(q_net, u_net, samples)

        # compute step loss and accuracy
        dataset_sizes += samples[0].size(0)
        batch_loss += loss.item() * samples[0].size(0)

    # compute epoch loss and relative error
    epoch_loss = batch_loss / dataset_sizes
    return (epoch_loss, err_u.item(), err_q.item())


def train(
    device,
    dataloader,
    q_net,
    u_net,
    loss_fn,
    pretrain_optimizer_u,
    optimizers,
    schedulers,
    pretrain_epochs_u,
    num_epochs,
    results_path,
    idx,
) -> None:
    for epoch in range(pretrain_epochs_u):
        start = time.time()
        _pretrain_u_epoch(device, dataloader, u_net, loss_fn, pretrain_optimizer_u)
        loss, err_u = _pretrain_validate_u_epoch(device, dataloader, u_net, loss_fn, idx)
        if device == 0 or str(device) in ['cpu', 'cuda', 'cuda:0']:
            print(f"Pre-Training for u: [{epoch + 1:>3d}/{pretrain_epochs_u:>3d}]", end=" ")
            print(f"spend time: [{(time.time() - start):6.2f} sec] ==", end=" ")
            print(f"loss = {loss:>2.4e}, rel err u = {err_u:>2.4e}")
    total_epochs = num_epochs[0] + num_epochs[1]
    results_mat = np.zeros((total_epochs, 3))
    for epoch in range(total_epochs):
        if epoch % 1000 == 0:
            _save_results(device, epoch, q_net, u_net, results_path)
        # switch to lbfgs optimizer after adam epochs
        optimizer = optimizers[0] if epoch < num_epochs[0] else optimizers[1]
        scheduler = schedulers[0] if epoch < num_epochs[0] else schedulers[1]
        start = time.time()
        _train_epoch(device, dataloader, q_net, u_net, loss_fn, optimizer, scheduler)
        loss_val, err_u, err_f = _validate_epoch(device, dataloader, q_net, u_net, loss_fn, idx)
        if device == 0 or str(device) in ['cpu', 'cuda', 'cuda:0']:
            results_mat[epoch, 0:3] = (loss_val, err_f, err_u)
            print(f"== Epochs: [{epoch + 1:>5d}/{total_epochs:>5d}] ==", end=" ")
            print(f"spend time: [{(time.time() - start):6.2f} sec] ==", end=" ")
            print(f"loss = {loss_val:>2.4e}, rel err q = {err_f:>2.4e}, rel err u = {err_u:>2.4e}")
    _save_results(device, total_epochs, q_net, u_net, results_path)
    np.savetxt(os.path.join(results_path, "loss.txt"), results_mat, delimiter=",")
