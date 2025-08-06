from typing import Callable

import torch
from torch import Tensor, autograd
import numpy as np
import os


def output_path(config, idx, noise_str: str):
    path = config["train_params"]["results_path"]
    if idx == "01":
        path = os.path.join(path, "one_peak")
    elif idx == "02":
        path = os.path.join(path, "non_smooth")
    elif idx == "03":
        path = os.path.join(path, "discontinuous")
    elif idx == "04":
        path = os.path.join(path, "one_peak_1d")
    else:
        raise ValueError(f"Unknown task idx: {idx}")

    lamb = config["loss_params"]["lamb"]
    if lamb == 0.0:
        path = os.path.join(path, f"result_00_{noise_str}")
    else:
        lamb = int(-np.log10(lamb))
        path = os.path.join(path, f"result_0{lamb}_{noise_str}")
    os.makedirs(path, exist_ok=True)
    config["train_params"]["results_path"] = path
    return config


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


def H2norm(func: Callable[[Tensor], Tensor], x: torch.Tensor) -> Tensor:
    x_clone = x.clone().detach().requires_grad_(True)
    fx = func(x_clone)
    result = torch.square(fx)
    grad = _gradient(fx, x_clone)
    result = torch.add(result, dot(grad, grad))
    partial_x1x1 = _gradient(grad, x_clone)
    result = torch.add(result, torch.square(partial_x1x1))
    # partial_x1x1 = _gradient(grad[:, 0:1], x_clone)[:, 0:1]
    # result = torch.add(result, torch.square(partial_x1x1))
    # partial_x2x2 = _gradient(grad[:, 1:2], x_clone)[:, 1:2]
    # result = torch.add(result, torch.square(partial_x2x2))
    # partial_x1x2 = _gradient(grad[:, 0:1], x_clone)[:, 1:2]
    # result = torch.add(result, 2.0 * torch.square(partial_x1x2))
    return result


def set_seed(rank=0, seed=42):
    seed += rank
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
