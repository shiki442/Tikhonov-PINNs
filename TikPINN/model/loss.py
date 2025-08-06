from torch import Tensor, mean, sqrt

from .problem import elliptic, neumann
from .utils import H2norm, ms, mse


def get_loss(alpha: float, lamb: float):
    return TikPINNLoss(alpha, lamb)


class TikPINNLoss(object):
    def __init__(self, alpha: float, lamb: float, d: int = 2) -> None:
        self.alpha = alpha
        self.lamb = lamb
        self.d = d

    @staticmethod
    def _measurement_loss(u, sample: Tensor) -> Tensor:
        ind = data_ind(sample.shape[1])
        interior, m_int = sample[:, ind['int']], sample[:, ind['m_int']]
        bdy, m_bdy = sample[:, ind['bdy']], sample[:, ind['m_bdy']]
        return mse(m_int, u(interior)) + mse(m_bdy, u(bdy))

    @staticmethod
    def _pinns_loss(q, u, sample: Tensor) -> Tensor:
        ind = data_ind(sample.shape[1])
        interior, f_val = sample[:, ind['int']], sample[:, ind['f_val']]
        loss_int = ms(elliptic(q, u, interior, f_val))
        bdy, normal, g_val = sample[:, ind['bdy']], sample[:, ind['normal']], sample[:, ind['g_val']]
        loss_neumann = mse(g_val, neumann(u, bdy, normal))
        return loss_int + loss_neumann

    @staticmethod
    def _regularization_loss(q, sample: Tensor) -> Tensor:
        ind = data_ind(sample.shape[1])
        interior = sample[:, ind['int']]
        return mean(H2norm(q, interior))

    def __call__(self, q, u, sample: Tensor) -> Tensor:
        return (
            self._measurement_loss(u, sample)
            + self.alpha * self._pinns_loss(q, u, sample)
            + self.lamb * self._regularization_loss(q, sample)
        )

    def measurement(self, u, sample: Tensor) -> Tensor:
        return self._measurement_loss(u, sample)


def relative_error_u(u, sample: Tensor) -> Tensor:
    ind = data_ind(sample.shape[1])
    interior = sample[:, ind['int']]
    u_dagger = sample[:, ind['u_dagger']]
    return sqrt(mse(u(interior), u_dagger) / ms(u_dagger))


def relative_error_q(q, sample: Tensor) -> Tensor:
    ind = data_ind(sample.shape[1])
    interior = sample[:, ind['int']]
    q_dagger = sample[:, ind['q_dagger']]
    return sqrt(mse(q(interior), q_dagger) / ms(q_dagger))


def data_ind(len_sample: int) -> dict:
    d = (len_sample - 6) // 3
    ind = dict()
    ind['int'] = [i for i in range(d)]
    ind['bdy'] = [i + d for i in range(d)]
    ind['normal'] = [i + 2 * d for i in range(d)]
    ind['m_int'] = [3 * d]
    ind['m_bdy'] = [3 * d + 1]
    ind['f_val'] = [3 * d + 2]
    ind['g_val'] = [3 * d + 3]
    ind['u_dagger'] = [3 * d + 4]
    ind['q_dagger'] = [3 * d + 5]
    return ind
