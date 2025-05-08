from torch import Tensor, mean, sqrt

from .problem import elliptic, neumann
from .utils import H2norm, ms, mse
import os
import numpy as np

def get_loss(alpha: float, lamb: float, idx: str, noise_str: str, data_path: str) -> object:
    return TikPINNLoss(alpha, lamb, idx, noise_str, data_path)


class TikPINNLoss(object):
    def __init__(self, alpha: float, lamb: float, idx: str, noise_str: str, data_path: str) -> None:
        self.alpha = alpha
        self.lamb = lamb
        data_path = os.path.join(data_path, "obs" + idx + "data" + noise_str + ".txt")
        data = np.loadtxt(data_path, dtype='float', delimiter=',')
        self.obs_data = Tensor(data).to("cuda:0")

    def _measurement_loss(self, u) -> Tensor:
        interior, m_int = self.obs_data[:, 0:2], self.obs_data[:, 4:5]
        bdy, m_bdy = self.obs_data[:, 2:4], self.obs_data[:, 5:6]
        return mse(m_int, u(interior)) + mse(m_bdy, u(bdy))

    @staticmethod
    def _pinns_loss(q, u, sample: Tensor) -> Tensor:
        interior, f_val = sample[:, 0:2], sample[:, 6:7]
        loss_int = ms(elliptic(q, u, interior, f_val))
        bdy, normal, g_val = sample[:, 2:4], sample[:, 4:6], sample[:, 7:8]
        loss_neumann = mse(g_val, neumann(u, bdy, normal))
        return loss_int + loss_neumann

    @staticmethod
    def _regularization_loss(q, sample: Tensor) -> Tensor:
        interior = sample[:, 0:2]
        return mean(H2norm(q, interior))

    def __call__(self, q, u, sample: Tensor) -> Tensor:
        return self._measurement_loss(u) + \
            self.alpha * self._pinns_loss(q, u, sample) + \
            self.lamb * self._regularization_loss(q, sample)
    
    def measurement(self, u) -> Tensor:
        return self._measurement_loss(u)


def relative_error_u(u, sample: Tensor) -> Tensor:
    interior = sample[:, 0:2]
    u_dagger = sample[:, 8:9]
    return sqrt(mse(u(interior), u_dagger) / ms(u_dagger))


def relative_error_q(q, sample: Tensor) -> Tensor:
    interior = sample[:, 0:2]
    q_dagger = sample[:, 9:10]
    return sqrt(mse(q(interior), q_dagger) / ms(q_dagger))
