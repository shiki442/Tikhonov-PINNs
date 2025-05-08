from torch import Tensor

from .utils import dot, grad, laplace


def elliptic(q, u, x: Tensor, f: Tensor) -> Tensor:
    return f + laplace(u, x) - q(x) * u(x)


def neumann(u, x: Tensor, normal: Tensor) -> Tensor:
    grad_u = grad(u, x)
    return dot(grad_u, normal)


def evaluate_q(q_net, x: Tensor) -> Tensor:
    q_net.eval()
    return q_net(x)


def evaluate_u(u_net, x: Tensor) -> Tensor:
    u_net.eval()
    return u_net(x)
