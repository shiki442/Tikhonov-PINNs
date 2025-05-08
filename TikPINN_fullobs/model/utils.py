from typing import Callable

import torch
from torch import Tensor, autograd


def dot(x, y):
    return torch.sum(x * y, dim=1, keepdim=True)


def ms(residual: Tensor) -> Tensor:
    return torch.mean(torch.square(residual))


def mse(pred: Tensor, target: Tensor) -> Tensor:
    return ms(torch.sub(pred, target))


def _gradient(outputs: Tensor, inputs: Tensor) -> Tensor:
    grad = autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(
        outputs), create_graph=True, only_inputs=True)
    return grad[0]


def grad(func: Callable[[Tensor], Tensor], x: torch.Tensor) -> Tensor:
    x_clone = x.clone().detach().requires_grad_(True)
    fx = func(x_clone)
    return _gradient(fx, x_clone)


def laplace(func: Callable[[Tensor], Tensor], x: torch.Tensor) -> Tensor:
    x_clone = x.clone().detach().requires_grad_(True)
    fx = func(x_clone)
    grad = _gradient(fx, x_clone)
    partial_x1x1 = _gradient(grad[:, 0:1], x_clone)[:, 0:1]
    partial_x2x2 = _gradient(grad[:, 1:2], x_clone)[:, 1:2]
    return torch.add(partial_x1x1, partial_x2x2)


def H2norm(func: Callable[[Tensor], Tensor], x: torch.Tensor) -> Tensor:
    x_clone = x.clone().detach().requires_grad_(True)
    fx = func(x_clone)
    result = torch.square(fx)
    grad = _gradient(fx, x_clone)
    result = torch.add(result, dot(grad, grad))
    partial_x1x1 = _gradient(grad[:, 0:1], x_clone)[:, 0:1]
    result = torch.add(result, torch.square(partial_x1x1))
    partial_x2x2 = _gradient(grad[:, 1:2], x_clone)[:, 1:2]
    result = torch.add(result, torch.square(partial_x2x2))
    partial_x1x2 = _gradient(grad[:, 0:1], x_clone)[:, 1:2]
    result = torch.add(result, 2.0 * torch.square(partial_x1x2))
    return result
