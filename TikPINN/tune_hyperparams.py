"""
Optuna hyperparameter optimization for TikPINN.
Run with: python tune_hyperparams.py --n_trials 50 --config_path config/params.yml
"""

import argparse
import os
import yaml
import optuna
import torch
import numpy as np
from optuna.samplers import TPESampler
# from optuna.storages import JournalStorage, JournalFileStorage

from main import get_problem_class, set_seed, check_config
from model.data import get_dataloader
from model.loss import get_loss
from model.nn import get_network
from model.optim import get_optimizer, get_scheduler
from model.utils import relative_error


def get_regular_grid(eval_points: int = 101, dim: int = 1, device: str = 'cpu') -> torch.Tensor:
    """Generate a regular grid for validation."""
    if dim == 1:
        x = torch.linspace(0, 1, eval_points, device=device).unsqueeze(1)
    elif dim == 2:
        x1 = torch.linspace(0, 1, eval_points, device=device)
        x2 = torch.linspace(0, 1, eval_points, device=device)
        xx1, xx2 = torch.meshgrid(x1, x2, indexing='ij')
        x = torch.stack([xx1.ravel(), xx2.ravel()], dim=1)
    else:
        raise ValueError(f"Only 1D and 2D problems are supported. Got dim={dim}")
    return x


def compute_errors_on_grid(device, q_net, u_net, problem, eval_points: int = 101):
    """Compute q and u relative errors on a regular grid."""
    q_net.eval()
    u_net.eval()

    x_grid = get_regular_grid(eval_points, problem.dim, device)

    with torch.no_grad():
        q_pred = q_net(x_grid)
        u_pred = u_net(x_grid)

    q_true_np = problem.q_dagger(x_grid.cpu().numpy())
    u_true_np = problem.u_dagger(x_grid.cpu().numpy())
    q_true = torch.from_numpy(q_true_np).to(device)
    u_true = torch.from_numpy(u_true_np).to(device)

    q_err = relative_error(q_pred, q_true)
    u_err = relative_error(u_pred, u_true)

    return q_err.item(), u_err.item()


def train_one_epoch(device, dataloader, q_net, u_net, loss_fn, optimizer, scheduler=None):
    """Train one epoch."""
    q_net.train()
    u_net.train()
    for _, samples in enumerate(dataloader):
        samples = (samples[0].to(device), samples[1].to(device))

        def closure():
            loss = loss_fn(q_net, u_net, samples)
            optimizer.zero_grad()
            loss.backward()
            return loss

        optimizer.step(closure)

    if scheduler is not None:
        scheduler.step()


def validate(device, dataloader, q_net, u_net, loss_fn):
    """Validate and return loss."""
    q_net.eval()
    u_net.eval()
    dataset_sizes = 0
    batch_loss = 0.0

    for _, samples in enumerate(dataloader):
        samples = (samples[0].to(device), samples[1].to(device))
        loss = loss_fn(q_net, u_net, samples)
        dataset_sizes += samples[0].size(0)
        batch_loss += loss.item() * samples[0].size(0)

    return batch_loss / dataset_sizes


def objective(trial, config, device):
    """
    Optuna objective function.

    Key hyperparameters to tune:
    - q_net: width, depth, activation
    - u_net: width, depth, activation
    - loss_params: alpha, lamb
    - optim_params: q_lr, u_lr
    - dataloader_params: batch_size
    """
    # Sample hyperparameters
    q_width = trial.suggest_int('q_width', 32, 256, step=32)
    q_depth = trial.suggest_int('q_depth', 2, 5)
    u_width = trial.suggest_int('u_width', 32, 256, step=32)
    u_depth = trial.suggest_int('u_depth', 2, 5)

    # Sample activation functions
    activation_choices = ['tanh', 'relu6p', 'swish']
    q_activation = trial.suggest_categorical('q_activation', activation_choices)
    u_activation = trial.suggest_categorical('u_activation', activation_choices)

    alpha = trial.suggest_float('alpha', 0.1, 10.0, log=True)
    lamb = trial.suggest_float('lamb', 1e-10, 1e-4, log=True)

    q_lr = trial.suggest_float('q_lr', 1e-4, 1e-2, log=True)
    u_lr = trial.suggest_float('u_lr', 1e-4, 1e-2, log=True)

    batch_size = trial.suggest_categorical('batch_size', [5000, 10000, 25000])

    # Modify config
    config = config.copy()
    config['q_net_params']['width'] = q_width
    config['q_net_params']['depth'] = q_depth
    config['q_net_params']['activation'] = q_activation
    config['u_net_params']['width'] = u_width
    config['u_net_params']['depth'] = u_depth
    config['u_net_params']['activation'] = u_activation
    config['loss_params']['alpha'] = alpha
    config['loss_params']['lamb'] = lamb
    config['optim_params_adam']['q_lr'] = q_lr
    config['optim_params_adam']['u_lr'] = u_lr
    config['dataloader_params']['batch_size'] = batch_size

    # Use a fixed seed for reproducibility
    seed = config.get('seed', 42)
    set_seed(rank=0, seed=seed)
    torch.manual_seed(seed)

    idx = config['task']['idx']
    noise_str = config['task']['noise_str']
    dim = config['task']['dim']
    check_config(config)

    # Create networks
    q_net = get_network(**config['q_net_params']).to(device)
    u_net = get_network(**config['u_net_params']).to(device)

    # Create dataloader
    dataloader = get_dataloader(idx=idx, noise_str=noise_str, **config['dataloader_params'])

    # Create loss and optimizers
    tikpinn_loss = get_loss(**config['loss_params'])
    optimizer = get_optimizer(q_net, u_net, **config['optim_params_adam'])
    scheduler = get_scheduler(optimizer, **config['scheduler_params'])

    # Training parameters
    pretrain_epochs = config['train_params'].get('pretrain_epochs_u', 10)
    adam_epochs = config['train_params']['num_epochs'][0]

    try:
        # Quick training (reduced epochs for tuning)
        for epoch in range(pretrain_epochs):
            q_net.train()
            u_net.train()
            for _, samples in enumerate(dataloader):
                samples = (samples[0].to(device), samples[1].to(device))
                loss = tikpinn_loss.measurement(u_net, samples)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        for epoch in range(adam_epochs):
            train_one_epoch(device, dataloader, q_net, u_net, tikpinn_loss, optimizer, scheduler)

        # Compute final error
        ProblemClass = get_problem_class(idx)
        problem = ProblemClass()
        q_err, u_err = compute_errors_on_grid(device, q_net, u_net, problem)

        # Objective: minimize weighted sum of errors
        objective_value = q_err + u_err

        # Report intermediate values for pruning
        trial.report(objective_value, step=0)

        # Handle pruning
        if trial.should_prune():
            raise optuna.TrialPruned()

        return objective_value

    except Exception as e:
        print(f"Trial failed with error: {e}")
        return float('inf')


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter tuning with Optuna')
    parser.add_argument('--config_path', type=str, default='config/params.yml', help='Path to config file')
    parser.add_argument('--n_trials', type=int, default=50, help='Number of Optuna trials')
    parser.add_argument('--n_jobs', type=int, default=1, help='Number of parallel jobs')
    parser.add_argument('--study_name', type=str, default='ex03', help='Study name for Optuna')
    parser.add_argument('--storage', type=str, default='sqlite:///study.db', help='Optuna storage (e.g., sqlite:///study.db)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    # Set global seed
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load config
    config_file_path = args.config_path
    with open(config_file_path, 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)

    # For tuning, reduce epochs to save time
    config['train_params']['num_epochs'] = [10000, 0]  # [Adam, LBFGS]
    config['train_params']['pretrain_epochs_u'] = 1000
    config['train_params']['heatmap_every_n_epochs'] = 0  # Disable heatmap logging
    config['checkpoint_params']['save_top_k'] = 0
    config['checkpoint_params']['save_last'] = False

    device = torch.device(args.device)
    print(f"Using device: {device}")
    print(f"Running {args.n_trials} trials...")

    # Create or load study
    sampler = TPESampler(seed=args.seed)
    study = optuna.create_study(
        study_name=args.study_name, sampler=sampler, direction='minimize', storage=args.storage, load_if_exists=True
    )

    # Run optimization
    study.optimize(lambda trial: objective(trial, config, device), n_trials=args.n_trials, n_jobs=args.n_jobs, show_progress_bar=True)

    # Print results
    print("\n" + "=" * 50)
    print("Best trial:")
    print(f"  Value (q_err + u_err): {study.best_trial.value:.6f}")
    print("  Params:")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")

    print("\n" + "=" * 50)
    print("Top 5 trials:")
    trials = sorted(study.trials, key=lambda t: t.value)[:5]
    for i, trial in enumerate(trials):
        print(f"  {i+1}. Value: {trial.value:.6f}, Params: {trial.params}")

    # Save best config
    best_config = config.copy()
    best_config['q_net_params']['width'] = study.best_trial.params['q_width']
    best_config['q_net_params']['depth'] = study.best_trial.params['q_depth']
    best_config['q_net_params']['activation'] = study.best_trial.params['q_activation']
    best_config['u_net_params']['width'] = study.best_trial.params['u_width']
    best_config['u_net_params']['depth'] = study.best_trial.params['u_depth']
    best_config['u_net_params']['activation'] = study.best_trial.params['u_activation']
    best_config['loss_params']['alpha'] = study.best_trial.params['alpha']
    best_config['loss_params']['lamb'] = study.best_trial.params['lamb']
    best_config['optim_params_adam']['q_lr'] = study.best_trial.params['q_lr']
    best_config['optim_params_adam']['u_lr'] = study.best_trial.params['u_lr']
    best_config['dataloader_params']['batch_size'] = study.best_trial.params['batch_size']

    # Restore original epochs for final training
    best_config['train_params']['num_epochs'] = [1000, 100]
    best_config['train_params']['pretrain_epochs_u'] = 100
    best_config['train_params']['heatmap_every_n_epochs'] = 100
    best_config['checkpoint_params']['save_top_k'] = 3
    best_config['checkpoint_params']['save_last'] = True

    # Save best config
    output_path = 'config/params_best.yml'
    with open(output_path, 'w') as f:
        yaml.dump(best_config, f, default_flow_style=False, sort_keys=False)
    print(f"\nBest config saved to: {output_path}")


if __name__ == '__main__':
    main()
