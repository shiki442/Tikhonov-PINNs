import os
import time

import numpy as np
import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter

from .utils import relative_error


def _plot_heatmap(data: np.ndarray, title: str, xlabel: str, ylabel: str, vmin: float = None, vmax: float = None) -> plt.Figure:
    """Create a heatmap figure."""
    fig, ax = plt.subplots(figsize=(6, 5))
    # Get extent from data shape: (rows, cols) -> (x_min, x_max, y_min, y_max)
    h, w = data.shape
    extent = [0, 1, 0, 1]
    im = ax.imshow(data, cmap='jet', origin='lower', aspect='auto', vmin=vmin, vmax=vmax, extent=extent)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.colorbar(im, ax=ax)
    return fig


def _log_heatmap(writer, tag: str, data: np.ndarray, step: int, shape: tuple = None) -> None:
    """
    Log a heatmap to TensorBoard.

    Args:
        writer: TensorBoard SummaryWriter
        tag: Name for the heatmap
        data: 2D numpy array or 1D array that will be reshaped
        step: Global step
        shape: Tuple of (height, width) to reshape 1D data
    """
    if shape is not None:
        data = data.reshape(shape)

    # Create a fake figure and close it
    fig = _plot_heatmap(data, tag, 'x1', 'x2')
    writer.add_figure(tag, fig, step)
    plt.close(fig)


def _get_regular_grid(eval_points: int = 101, dim: int = 1, device: str = 'cpu') -> torch.Tensor:
    """
    Generate a regular grid for validation.

    Args:
        eval_points: Number of evaluation points (per dimension for multi-D)
        dim: Dimension of the problem
        device: Device to place the tensor

    Returns:
        Regular grid tensor of shape (n_points, dim)
    """
    if dim == 1:
        x = torch.linspace(0, 1, eval_points, device=device).unsqueeze(1)
    elif dim == 2:
        # Create a 2D grid
        x1 = torch.linspace(0, 1, eval_points, device=device)
        x2 = torch.linspace(0, 1, eval_points, device=device)
        xx1, xx2 = torch.meshgrid(x1, x2, indexing='ij')
        x = torch.stack([xx1.ravel(), xx2.ravel()], dim=1)
    else:
        raise ValueError(f"Only 1D and 2D problems are supported. Got dim={dim}")
    return x


def _compute_errors_on_grid(device, q_net, u_net, problem, eval_points: int = 101) -> tuple:
    """
    Compute q and u relative errors on a regular grid.

    Args:
        device: Device to compute on
        q_net: q network
        u_net: u network
        problem: ProblemND instance providing q_dagger and u_dagger
        eval_points: Number of evaluation points

    Returns:
        Tuple of (q_err, u_err)
    """
    q_net.eval()
    u_net.eval()

    # Generate regular grid
    x_grid = _get_regular_grid(eval_points, problem.dim, device)

    # Compute predictions
    with torch.no_grad():
        q_pred = q_net(x_grid)
        u_pred = u_net(x_grid)

    # Get true solutions (convert numpy to torch)
    q_true_np = problem.q_dagger(x_grid.cpu().numpy())
    u_true_np = problem.u_dagger(x_grid.cpu().numpy())
    q_true = torch.from_numpy(q_true_np).to(device)
    u_true = torch.from_numpy(u_true_np).to(device)

    # Compute relative errors
    q_err = relative_error(q_pred, q_true)
    u_err = relative_error(u_pred, u_true)

    return q_err.item(), u_err.item()


def _get_predictions_on_grid(device, q_net, u_net, problem, eval_points: int = 101) -> dict:
    """
    Get predictions and ground truth on a regular grid.

    Args:
        device: Device to compute on
        q_net: q network
        u_net: u network
        problem: ProblemND instance providing q_dagger and u_dagger
        eval_points: Number of evaluation points

    Returns:
        Dictionary with 'q_pred', 'u_pred', 'q_true', 'u_true', 'shape'
    """
    q_net.eval()
    u_net.eval()

    # Generate regular grid
    x_grid = _get_regular_grid(eval_points, problem.dim, device)

    # Compute predictions
    with torch.no_grad():
        q_pred = q_net(x_grid)
        u_pred = u_net(x_grid)

    # Get true solutions
    q_true_np = problem.q_dagger(x_grid.cpu().numpy())
    u_true_np = problem.u_dagger(x_grid.cpu().numpy())
    q_true = torch.from_numpy(q_true_np).to(device)
    u_true = torch.from_numpy(u_true_np).to(device)

    # Get shape for heatmap
    if problem.dim == 1:
        shape = (eval_points,)
    elif problem.dim == 2:
        shape = (eval_points, eval_points)
    else:
        raise ValueError(f"Only 1D and 2D problems are supported. Got dim={problem.dim}")

    return {
        'q_pred': q_pred.cpu().numpy().ravel(),
        'u_pred': u_pred.cpu().numpy().ravel(),
        'q_true': q_true.cpu().numpy().ravel(),
        'u_true': u_true.cpu().numpy().ravel(),
        'shape': shape,
    }


def _log_heatmaps(writer, data: dict, step: int, prefix: str = '') -> None:
    """
    Log q and u heatmaps to TensorBoard.

    Args:
        writer: TensorBoard SummaryWriter
        data: Dictionary with predictions and ground truth
        step: Global step
        prefix: Prefix for tag names
    """
    shape = data['shape']

    # Log predictions
    if prefix:
        q_tag = f'{prefix}/q_pred'
        u_tag = f'{prefix}/u_pred'
    else:
        q_tag = 'q_pred'
        u_tag = 'u_pred'

    # Get global vmin/vmax for consistent color scale
    q_vmin, q_vmax = data['q_true'].min(), data['q_true'].max()
    u_vmin, u_vmax = data['u_true'].min(), data['u_true'].max()

    # Log ground truth
    fig_q = _plot_heatmap(data['q_true'].reshape(shape), 'Ground Truth (q)', 'x1', 'x2', q_vmin, q_vmax)
    fig_u = _plot_heatmap(data['u_true'].reshape(shape), 'Ground Truth (u)', 'x1', 'x2', u_vmin, u_vmax)
    writer.add_figure(f'{prefix}/q_true' if prefix else 'q_true', fig_q, step)
    writer.add_figure(f'{prefix}/u_true' if prefix else 'u_true', fig_u, step)
    plt.close(fig_q)
    plt.close(fig_u)

    # Log predictions
    fig_q = _plot_heatmap(data['q_pred'].reshape(shape), 'Prediction (q)', 'x1', 'x2', q_vmin, q_vmax)
    fig_u = _plot_heatmap(data['u_pred'].reshape(shape), 'Prediction (u)', 'x1', 'x2', u_vmin, u_vmax)
    writer.add_figure(q_tag, fig_q, step)
    writer.add_figure(u_tag, fig_u, step)
    plt.close(fig_q)
    plt.close(fig_u)


def _save_checkpoint(device, epoch, q_net, u_net, optimizer, scheduler, results_path, filename='checkpoint.pt'):
    """Save model checkpoint for resuming training."""
    # Get state dicts from DDP wrapped models if applicable
    q_net_state = q_net.module.state_dict() if hasattr(q_net, 'module') else q_net.state_dict()
    u_net_state = u_net.module.state_dict() if hasattr(u_net, 'module') else u_net.state_dict()

    checkpoint = {
        'epoch': epoch,
        'q_net': q_net_state,
        'u_net': u_net_state,
        'optimizer': optimizer.state_dict() if optimizer is not None else None,
        'scheduler': scheduler.state_dict() if scheduler is not None else None,
    }
    torch.save(checkpoint, os.path.join(results_path, filename))


def _load_checkpoint(checkpoint_path, q_net, u_net, optimizer=None, scheduler=None, device='cpu'):
    """Load checkpoint and return training state."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Handle DDP wrapped models
    q_net_state = q_net.module.state_dict() if hasattr(q_net, 'module') else q_net.state_dict()
    u_net_state = u_net.module.state_dict() if hasattr(u_net, 'module') else u_net.state_dict()

    q_net_state.update(checkpoint['q_net'])
    u_net_state.update(checkpoint['u_net'])

    if hasattr(q_net, 'module'):
        q_net.module.load_state_dict(q_net_state)
    else:
        q_net.load_state_dict(q_net_state)

    if hasattr(u_net, 'module'):
        u_net.module.load_state_dict(u_net_state)
    else:
        u_net.load_state_dict(u_net_state)

    if optimizer is not None and checkpoint['optimizer'] is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])

    if scheduler is not None and checkpoint['scheduler'] is not None:
        scheduler.load_state_dict(checkpoint['scheduler'])

    return checkpoint['epoch']


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


def _pretrain_validate_u_epoch(device, dataloader, u_net, loss_fn, writer, global_step, problem, eval_points=101) -> float:
    """Validate"""
    u_net.eval()
    dataset_sizes = 0
    batch_loss = 0.0

    for _, samples in enumerate(dataloader):
        # load data to CUDA - samples is now (int_sample, bdy_sample) tuple
        samples = (samples[0].to(device), samples[1].to(device))
        # forward and compute loss
        loss = loss_fn.measurement(u_net, samples)
        # compute step loss and accuracy
        dataset_sizes += samples[0].size(0)
        batch_loss += loss.item() * samples[0].size(0)
    # compute epoch loss
    epoch_loss = batch_loss / dataset_sizes

    # Compute u error on regular grid and log to TensorBoard
    q_net_dummy = u_net  # pass u_net as placeholder, only u_net is used in pretrain
    q_err, u_err = _compute_errors_on_grid(device, q_net_dummy, u_net, problem, eval_points)
    if writer is not None:
        writer.add_scalar('Loss/pretrain_u', epoch_loss, global_step)
        writer.add_scalar('Loss/measurement', epoch_loss, global_step)
        writer.add_scalar('Error/u_relative', u_err, global_step)

    return epoch_loss, q_err, u_err


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


def _validate_epoch(device, dataloader, q_net, u_net, loss_fn, writer, global_step, problem, eval_points=101) -> float:
    """Validate"""
    q_net.eval()
    u_net.eval()
    dataset_sizes = 0
    batch_loss = 0.0
    batch_loss_components = {'measurement': 0.0, 'pinns': 0.0, 'regularization': 0.0}

    for _, samples in enumerate(dataloader):
        # load data to CUDA - samples is now (int_sample, bdy_sample) tuple
        samples = (samples[0].to(device), samples[1].to(device))
        # forward and compute loss
        loss_components = loss_fn.get_loss_components(q_net, u_net, samples)
        loss = loss_components['total']

        # compute step loss and accuracy
        dataset_sizes += samples[0].size(0)
        batch_loss += loss.item() * samples[0].size(0)
        batch_loss_components['measurement'] += loss_components['measurement'] * samples[0].size(0)
        batch_loss_components['pinns'] += loss_components['pinns'] * samples[0].size(0)
        batch_loss_components['regularization'] += loss_components['regularization'] * samples[0].size(0)

    # compute epoch loss
    epoch_loss = batch_loss / dataset_sizes
    epoch_loss_components = {
        'measurement': batch_loss_components['measurement'] / dataset_sizes,
        'pinns': batch_loss_components['pinns'] / dataset_sizes,
        'regularization': batch_loss_components['regularization'] / dataset_sizes,
    }

    # Compute q and u errors on regular grid and log to TensorBoard
    q_err, u_err = _compute_errors_on_grid(device, q_net, u_net, problem, eval_points)
    if writer is not None:
        writer.add_scalar('Loss/joint', epoch_loss, global_step)
        writer.add_scalar('Loss/measurement', epoch_loss_components['measurement'], global_step)
        writer.add_scalar('Loss/pinns', epoch_loss_components['pinns'], global_step)
        writer.add_scalar('Loss/regularization', epoch_loss_components['regularization'], global_step)
        writer.add_scalar('Error/q_relative', q_err, global_step)
        writer.add_scalar('Error/u_relative', u_err, global_step)

    return epoch_loss, q_err, u_err


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
    problem,
    writer=None,
    eval_points=101,
    checkpoint_path=None,
    ckpt_every_n_epochs=100,
    ckpt_save_last=True,
    heatmap_every_n_epochs=100,
) -> None:
    """
    Training function with checkpoint support.

    Args:
        problem: ProblemND instance providing q_dagger and u_dagger for error computation
        writer: TensorBoard SummaryWriter instance
        eval_points: Number of evaluation points on regular grid
        checkpoint_path: Path to checkpoint file for resuming. If None, start from scratch.
                        If file exists, training resumes from the saved epoch.
        ckpt_every_n_epochs: Save checkpoint every N epochs
        ckpt_save_last: Save final checkpoint at the end
        heatmap_every_n_epochs: Log heatmaps every N epochs (0 to disable)
    """
    start_epoch = 0

    # Try to load checkpoint if path provided
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        start_epoch = _load_checkpoint(checkpoint_path, q_net, u_net, pretrain_optimizer_u, None, device)
        print(f"Resumed from epoch {start_epoch}")

    # Log ground truth heatmaps at the beginning (only on main process)
    if writer is not None and (device == 0 or str(device) in ['cpu', 'cuda', 'cuda:0']):
        print("Logging ground truth heatmaps...")
        grid_data = _get_predictions_on_grid(device, q_net, u_net, problem, eval_points)
        # Only use true values for ground truth
        ground_truth_data = {
            'q_pred': grid_data['q_true'],  # Use true as placeholder
            'u_pred': grid_data['u_true'],
            'q_true': grid_data['q_true'],
            'u_true': grid_data['u_true'],
            'shape': grid_data['shape'],
        }
        _log_heatmaps(writer, ground_truth_data, step=0, prefix='GroundTruth')

    # Log initial network predictions at the beginning
    if writer is not None and (device == 0 or str(device) in ['cpu', 'cuda', 'cuda:0']) and heatmap_every_n_epochs > 0:
        print("Logging initial network predictions...")
        grid_data = _get_predictions_on_grid(device, q_net, u_net, problem, eval_points)
        _log_heatmaps(writer, grid_data, step=0, prefix='Init')

    # Pretraining phase for u_net
    for epoch in range(start_epoch, pretrain_epochs_u):
        start = time.time()
        _pretrain_u_epoch(device, dataloader, u_net, loss_fn, pretrain_optimizer_u)
        loss, q_err, u_err = _pretrain_validate_u_epoch(device, dataloader, u_net, loss_fn, writer, epoch, problem, eval_points)
        if device == 0 or str(device) in ['cpu', 'cuda', 'cuda:0']:
            print(f"Pre-Training for u: [{epoch + 1:>3d}/{pretrain_epochs_u:>3d}]", end=" ")
            print(f"spend time: [{(time.time() - start):6.2f} sec] ==", end=" ")
            print(f"loss = {loss:>2.4e}, u_err = {u_err:>2.4e}")

        # Log heatmaps during pretraining
        if writer is not None and heatmap_every_n_epochs > 0 and (epoch + 1) % heatmap_every_n_epochs == 0:
            if device == 0 or str(device) in ['cpu', 'cuda', 'cuda:0']:
                grid_data = _get_predictions_on_grid(device, q_net, u_net, problem, eval_points)
                _log_heatmaps(writer, grid_data, step=epoch + 1, prefix='Pretrain')

        # Save checkpoint every N epochs during pretraining
        if ckpt_every_n_epochs > 0 and (epoch + 1) % ckpt_every_n_epochs == 0:
            _save_checkpoint(device, epoch + 1, q_net, u_net, pretrain_optimizer_u, None, results_path, f'checkpoint_pretrain_{epoch+1}.pt')

    # Save checkpoint after pretraining
    _save_checkpoint(device, pretrain_epochs_u, q_net, u_net, pretrain_optimizer_u, None, results_path, 'checkpoint_pretrain_final.pt')

    # Log heatmap after pretraining
    if writer is not None and heatmap_every_n_epochs > 0:
        if device == 0 or str(device) in ['cpu', 'cuda', 'cuda:0']:
            grid_data = _get_predictions_on_grid(device, q_net, u_net, problem, eval_points)
            _log_heatmaps(writer, grid_data, step=pretrain_epochs_u, prefix='AfterPretrain')

    total_epochs = num_epochs[0] + num_epochs[1]

    # Joint training phase for u_net and q_net
    for epoch in range(total_epochs):
        # switch to lbfgs optimizer after adam epochs
        optimizer = optimizers[0] if epoch < num_epochs[0] else optimizers[1]
        scheduler = schedulers[0] if epoch < num_epochs[0] else schedulers[1]
        start = time.time()
        _train_epoch(device, dataloader, q_net, u_net, loss_fn, optimizer, scheduler)
        loss_val, q_err, u_err = _validate_epoch(
            device, dataloader, q_net, u_net, loss_fn, writer, pretrain_epochs_u + epoch, problem, eval_points
        )
        if device == 0 or str(device) in ['cpu', 'cuda', 'cuda:0']:
            print(f"== Epochs: [{epoch + 1:>5d}/{total_epochs:>5d}] ==", end=" ")
            print(f"spend time: [{(time.time() - start):6.2f} sec] ==", end=" ")
            print(f"loss = {loss_val:>2.4e}, q_err = {q_err:>2.4e}, u_err = {u_err:>2.4e}")

        # Log heatmaps during joint training
        if writer is not None and heatmap_every_n_epochs > 0 and (epoch + 1) % heatmap_every_n_epochs == 0:
            if device == 0 or str(device) in ['cpu', 'cuda', 'cuda:0']:
                grid_data = _get_predictions_on_grid(device, q_net, u_net, problem, eval_points)
                _log_heatmaps(writer, grid_data, step=pretrain_epochs_u + epoch + 1, prefix='Train')

        # Save checkpoint every N epochs
        if ckpt_every_n_epochs > 0 and (epoch + 1) % ckpt_every_n_epochs == 0:
            _save_checkpoint(
                device,
                pretrain_epochs_u + epoch + 1,
                q_net,
                u_net,
                optimizer,
                scheduler,
                results_path,
                f'checkpoint_{pretrain_epochs_u + epoch + 1}.pt',
            )

    # Log final heatmaps at the end of training
    if writer is not None and (device == 0 or str(device) in ['cpu', 'cuda', 'cuda:0']):
        print("Logging final heatmaps...")
        grid_data = _get_predictions_on_grid(device, q_net, u_net, problem, eval_points)
        _log_heatmaps(writer, grid_data, step=pretrain_epochs_u + total_epochs, prefix='Final')

    # Save final checkpoint
    if ckpt_save_last:
        _save_checkpoint(device, pretrain_epochs_u + total_epochs, q_net, u_net, optimizer, scheduler, results_path, 'checkpoint_final.pt')
