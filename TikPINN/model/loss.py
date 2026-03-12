import torch
from torch import Tensor, mean, sqrt
from typing import Tuple
from .problem import elliptic, neumann
from .utils import H2norm, L2norm, ms, mse


def get_loss(alpha: float, lamb: float, regularization: str = 'H2') -> object:
    return TikPINNLoss(alpha, lamb, regularization)


class TikPINNLoss(object):
    def __init__(self, alpha: float, lamb: float, regularization: str) -> None:
        self.alpha = alpha
        self.lamb = lamb
        self.regularization = regularization

    @staticmethod
    def _measurement_loss(u, samples: Tuple[Tensor, Tensor]) -> Tensor:
        """
        Compute measurement loss.

        Args:
            u: The u_net model
            samples: Tuple of (int_sample, bdy_sample)
                - int_sample: [int_x, m_int, f_val, u_dagger, q_dagger]
                - bdy_sample: [bdy_x, normal, m_bdy, g_val]
        """
        int_sample, bdy_sample = samples
        # int_sample columns: [int_x1..xd, m_int, f_val, u_dagger, q_dagger]
        # bdy_sample columns: [bdy_x1..xd, normal_x1..xd, m_bdy, g_val]
        d = (int_sample.shape[1] - 4)  # dimension = total_cols - 4 scalar cols

        interior = int_sample[:, :d]
        m_int = int_sample[:, d:d+1]

        bdy = bdy_sample[:, :d]
        m_bdy = bdy_sample[:, 2*d:2*d+1]

        return mse(m_int, u(interior)) + mse(m_bdy, u(bdy))

    @staticmethod
    def _pinns_loss(q, u, samples: Tuple[Tensor, Tensor]) -> Tensor:
        """
        Compute PINNs loss (PDE residual + boundary condition).

        Args:
            q: The q_net model
            u: The u_net model
            samples: Tuple of (int_sample, bdy_sample)
        """
        int_sample, bdy_sample = samples
        d = (int_sample.shape[1] - 4)

        # Interior: compute PDE residual
        interior = int_sample[:, :d]
        f_val = int_sample[:, d+1:d+2]
        loss_int = ms(elliptic(q, u, interior, f_val))

        # Boundary: compute Neumann condition
        bdy = bdy_sample[:, :d]
        normal = bdy_sample[:, d:2*d]
        g_val = bdy_sample[:, 2*d+1:2*d+2]
        loss_neumann = mse(g_val, neumann(u, bdy, normal))

        return loss_int + loss_neumann

    def _regularization_loss(self, q, samples: Tuple[Tensor, Tensor]) -> Tensor:
        """
        Compute Tikhonov regularization loss on q.

        Args:
            q: The q_net model
            samples: Tuple of (int_sample, bdy_sample)
        """
        int_sample, _ = samples
        d = (int_sample.shape[1] - 4)
        interior = int_sample[:, :d]

        if self.regularization == 'H2':
            return mean(H2norm(q, interior))
        elif self.regularization == 'L2':
            return mean(L2norm(q, interior))
        else:
            return torch.tensor([0.0], device=int_sample.device)

    def __call__(self, q, u, samples: Tuple[Tensor, Tensor]) -> Tensor:
        """
        Compute total TikPINN loss.

        Args:
            q: The q_net model
            u: The u_net model
            samples: Tuple of (int_sample, bdy_sample)
        """
        return (
            self._measurement_loss(u, samples)
            + self.alpha * self._pinns_loss(q, u, samples)
            + self.lamb * self._regularization_loss(q, samples)
        )

    def measurement(self, u, samples: Tuple[Tensor, Tensor]) -> Tensor:
        """
        Compute measurement loss only (for pre-training u_net).

        Args:
            u: The u_net model
            samples: Tuple of (int_sample, bdy_sample)
        """
        return self._measurement_loss(u, samples)


def relative_error_u(u, samples: Tuple[Tensor, Tensor]) -> Tensor:
    """
    Compute relative error for u.

    Args:
        u: The u_net model
        samples: Tuple of (int_sample, bdy_sample)
    """
    int_sample, _ = samples
    d = (int_sample.shape[1] - 4)
    interior = int_sample[:, :d]
    u_dagger = int_sample[:, d+2:d+3]
    return sqrt(mse(u(interior), u_dagger) / ms(u_dagger))


def relative_error_q(q, samples: Tuple[Tensor, Tensor]) -> Tensor:
    """
    Compute relative error for q.

    Args:
        q: The q_net model
        samples: Tuple of (int_sample, bdy_sample)
    """
    int_sample, _ = samples
    d = (int_sample.shape[1] - 4)
    interior = int_sample[:, :d]
    q_dagger = int_sample[:, d+3:d+4]
    return sqrt(mse(q(interior), q_dagger) / ms(q_dagger))
