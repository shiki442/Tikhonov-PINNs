from typing import Callable

import torch
from torch import Tensor, autograd
import numpy as np
import os
import warnings


def check_config(config: dict) -> dict:
    # check and create results path
    idx = config['task']['idx']
    noise_str = config['task']['noise_str']
    path = config["train_params"]["results_path"]
    if idx == "01":
        path = os.path.join(path, "one_peak")
    elif idx == "02":
        path = os.path.join(path, "non_smooth")
    elif idx == "03":
        path = os.path.join(path, "discontinuous")
    elif idx == "04":
        path = os.path.join(path, "one_peak_1d_w")
    elif idx == "05":
        path = os.path.join(path, "one_peak_1d_w")
    elif idx == "06":
        path = os.path.join(path, "two_peaks_1d")
    else:
        raise ValueError(f"Unknown task idx: {idx}")

    config["train_params"]["idx"] = config['task']['idx']

    if config["dataloader_params"]["n_samples"] < config["dataloader_params"]["batch_size"]:
        warnings.warn(
            f"n_samples {config['dataloader_params']['n_samples']} is less than batch_size {config['dataloader_params']['batch_size']}. "
        )
        config["dataloader_params"]["batch_size"] = config["dataloader_params"]["n_samples"]

    lamb = config["loss_params"]["lamb"]
    # 关于lamb和噪声
    # if lamb == 0.0:
    #     path = os.path.join(path, f"result_00_{noise_str}")
    # else:
    #     lamb = int(-np.log10(lamb))
    #     path = os.path.join(path, f"result_0{lamb}_{noise_str}")
    # 关于宽度
    seed = config["seed"]
    width = config["q_net_params"]["width_list"][0]
    path = os.path.join(path, f"result_{width}_{seed}")
    # 关于样本量和种子
    # seed = config["seed"]
    # n = config["dataloader_params"]["n_samples"]
    # path = os.path.join(path, f"result_{n}_{seed}")
    # 关于正则化和种子
    # lamb = config["loss_params"]["lamb"]
    # reg = config["loss_params"]["regularization"]
    # lamb = int(-np.log10(lamb)) if lamb > 0 else 0
    # path = os.path.join(path, f"result_{reg}_0{lamb}")

    os.makedirs(path, exist_ok=True)
    config["train_params"]["results_path"] = path

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
