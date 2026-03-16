from typing import Callable

import torch
from torch import Tensor, autograd
import numpy as np
import os
import warnings
import yaml


def save_config(config: dict, path: str, filename: str = 'config.yaml'):
    """
    Save configuration to YAML file.

    Args:
        config: Configuration dictionary
        path: Directory path to save the config
        filename: Output filename (default: 'config.yaml')
    """
    os.makedirs(path, exist_ok=True)
    output_path = os.path.join(path, filename)
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Config saved to: {output_path}")


def get_next_version(path: str) -> str:
    """
    Get the next available version folder name (v1, v2, v3, ...).

    Args:
        path: Base path to check for existing versions

    Returns:
        Next version folder name (e.g., 'v1', 'v2', etc.)
    """
    existing_versions = []
    if os.path.exists(path):
        for name in os.listdir(path):
            if name.startswith('v') and os.path.isdir(os.path.join(path, name)):
                try:
                    version_num = int(name[1:])
                    existing_versions.append(version_num)
                except ValueError:
                    pass

    if existing_versions:
        next_version = max(existing_versions) + 1
    else:
        next_version = 1

    return f'v{next_version}'


def check_config(config: dict, config_file_path: str = None) -> dict:
    """
    Validate config and set up results path with version management.

    Args:
        config: Configuration dictionary
        config_file_path: Path to the original config YAML file (optional)

    Returns:
        Updated configuration dictionary with results_path set
    """
    # check and create results path
    idx = config['task']['idx']
    base_path = config["train_params"]["logs_path"]
    base_path = os.path.join(base_path, f"ex{idx}/")

    if config["dataloader_params"]["n_samples"] < config["dataloader_params"]["batch_size"]:
        warnings.warn(
            f"n_samples {config['dataloader_params']['n_samples']} is less than batch_size {config['dataloader_params']['batch_size']}. "
        )
        config["dataloader_params"]["batch_size"] = config["dataloader_params"]["n_samples"]

    # Create base path if not exists
    os.makedirs(base_path, exist_ok=True)

    # Get next version folder
    version = get_next_version(base_path)
    path = os.path.join(base_path, version)
    os.makedirs(path, exist_ok=True)

    config["train_params"]["results_path"] = path
    config["train_params"]["version"] = version

    # Save config file to results folder
    if config_file_path is not None and os.path.exists(config_file_path):
        import shutil
        config_save_path = os.path.join(path, 'config.yaml')
        shutil.copy(config_file_path, config_save_path)
        print(f"Config copied to: {config_save_path}")
    else:
        # Save config directly if no source file provided
        save_config(config, path)

    print(f"Results will be saved to: {path}")

    # check in_features
    config['q_net_params']['in_features'] = config['task']['dim']
    config['u_net_params']['in_features'] = config['task']['dim']

    # check total epochs
    config['scheduler_params']['total_steps'] = config['train_params']['num_epochs'][0] + config['train_params']['num_epochs'][1]

    return config


def q_dagger(x, idx=None):
    if idx == "01":
        return torch.exp(x)
    elif idx == "06":
        return (1.0 + x) * (torch.exp(x))
    else:
        raise ValueError(f"Unknown task idx: {idx}")


def u_dagger(x, idx=None):
    if idx == "01":
        return 1.0 + torch.sin(torch.pi * x)
    elif idx == "06":
        return 1.0 + torch.sin(torch.pi * x)
    else:
        raise ValueError(f"Unknown task idx: {idx}")


def relative_error(solution, solution_ref):
    err = solution - solution_ref
    relative_err = torch.linalg.norm(err) / torch.linalg.norm(solution_ref)
    return relative_err


def dot(x, y):
    return torch.sum(x * y, dim=1, keepdim=True)


def ms(residual: Tensor) -> Tensor:
    return torch.mean(torch.square(residual))


def mse(pred: Tensor, target: Tensor) -> Tensor:
    return ms(torch.sub(pred, target))


def _gradient(outputs: Tensor, inputs: Tensor) -> Tensor:
    grad = autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True, only_inputs=True)
    return grad[0]


def grad(func: Callable[[Tensor], Tensor], x: torch.Tensor) -> Tensor:
    x_clone = x.clone().detach().requires_grad_(True)
    fx = func(x_clone)
    return _gradient(fx, x_clone)


def laplace(func: Callable[[Tensor], Tensor], x: torch.Tensor) -> Tensor:
    x_clone = x.clone().detach().requires_grad_(True)
    fx = func(x_clone)
    grad = _gradient(fx, x_clone)
    lap = torch.zeros_like(x)
    for i in range(x_clone.shape[1]):
        second_grad = _gradient(grad[:, i : i + 1], x_clone)
        lap = torch.add(lap, second_grad[:, i : i + 1])
    return lap


def L2norm(func: Callable[[Tensor], Tensor], x: torch.Tensor) -> Tensor:
    d = x.shape[1]
    x_clone = x.clone().detach().requires_grad_(True)
    fx = func(x_clone)
    result = torch.square(fx)
    return result


def H2norm(func: Callable[[Tensor], Tensor], x: torch.Tensor) -> Tensor:
    d = x.shape[1]
    x_clone = x.clone().detach().requires_grad_(True)
    fx = func(x_clone)
    result = torch.square(fx)
    grad = _gradient(fx, x_clone)
    result = result + dot(grad, grad)
    D2x = torch.zeros_like(result)
    for i in range(d):
        grad_i = _gradient(grad[:, i : i + 1], x_clone)
        D2x = D2x + torch.sum(torch.square(grad_i), dim=-1).unsqueeze(-1)
    result = result + D2x
    return result


def set_seed(rank=0, seed=42):
    seed += rank
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
