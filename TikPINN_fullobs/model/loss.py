from torch import Tensor, mean, sqrt

from .problem import elliptic, neumann
from .utils import H2norm, ms, mse


def get_loss(alpha: float, lamb: float):
    return TikPINNLoss(alpha, lamb)


class TikPINNLoss(object):
    def __init__(self, alpha: float, lamb: float) -> None:
        self.alpha = alpha
        self.lamb = lamb

    @staticmethod
    def _measurement_loss(u, sample: Tensor) -> Tensor:
        interior, m_int = sample[:, 0:2], sample[:, 6:7]
        bdy, m_bdy = sample[:, 2:4], sample[:, 7:8]
        return mse(m_int, u(interior)) + mse(m_bdy, u(bdy))

    @staticmethod
    def _pinns_loss(q, u, sample: Tensor) -> Tensor:
        interior, f_val = sample[:, 0:2], sample[:, 8:9]
        loss_int = ms(elliptic(q, u, interior, f_val))
        bdy, normal, g_val = sample[:, 2:4], sample[:, 4:6], sample[:, 9:10]
        loss_neumann = mse(g_val, neumann(u, bdy, normal))
        return loss_int + loss_neumann

    @staticmethod
    def _regularization_loss(q, sample: Tensor) -> Tensor:
        interior = sample[:, 0:2]
        return mean(H2norm(q, interior))

    def __call__(self, q, u, sample: Tensor) -> Tensor:
        return self._measurement_loss(u, sample) + \
            self.alpha * self._pinns_loss(q, u, sample) + \
            self.lamb * self._regularization_loss(q, sample)
    
    def measurement(self, u, sample: Tensor) -> Tensor:
        return self._measurement_loss(u, sample)


def relative_error_u(u, sample: Tensor) -> Tensor:
    interior = sample[:, 0:2]
    u_dagger = sample[:, 10:11]
    return sqrt(mse(u(interior), u_dagger) / ms(u_dagger))


def relative_error_q(q, sample: Tensor) -> Tensor:
    interior = sample[:, 0:2]
    q_dagger = sample[:, 11:12]
    return sqrt(mse(q(interior), q_dagger) / ms(q_dagger))
