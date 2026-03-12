"""
Example 02: 2D sine product with truncated Gaussian peak in q.

q(x,y) = 0.75 + clamp(0.25, 0.75, 2.0*exp(-25*((x-0.6)² + (y-0.4)²)))
u(x,y) = 1.0 + sin(π*x)*sin(π*y)
"""

import numpy as np
from .problem_base_nd import ProblemND


class Example02Problem(ProblemND):
    """
    Example 02: 2D inverse problem with truncated Gaussian peak in q.

    Problem definition:
        tmp(x, y) = 2.0 * exp(-25*((x-0.6)² + (y-0.4)²))
        q_dagger(x, y) = 0.75 + clamp(0.25, 0.75, tmp(x,y))
        u_dagger(x, y) = 1.0 + sin(π*x)*sin(π*y)

    where clamp(min, max, val) = max(min, min(max, val))

    Derivatives (same as example01 since u is the same):
        ∂u/∂x = π*cos(π*x)*sin(π*y)
        ∂u/∂y = π*sin(π*x)*cos(π*y)
        Δu = -2*π²*sin(π*x)*sin(π*y)
    """

    @property
    def name(self) -> str:
        return "example02"

    @property
    def dim(self) -> int:
        return 2

    def q_dagger(self, x: np.ndarray) -> np.ndarray:
        """
        True parameter q(x,y) = 0.75 + clamp(0.25, 0.75, tmp(x,y)).

        tmp(x,y) = 2.0 * exp(-25*((x-0.6)² + (y-0.4)²))
        """
        x1, x2 = x[:, 0:1], x[:, 1:2]
        tmp = 2.0 * np.exp(-25.0 * ((x1 - 0.6)**2 + (x2 - 0.4)**2))
        # clamp between 0.25 and 0.75
        tmp_clamped = np.maximum(0.25, np.minimum(0.75, tmp))
        return 0.75 + tmp_clamped

    def u_dagger(self, x: np.ndarray) -> np.ndarray:
        """True solution u(x,y) = 1.0 + sin(π*x)*sin(π*y)."""
        x1, x2 = x[:, 0:1], x[:, 1:2]
        return 1.0 + np.sin(np.pi * x1) * np.sin(np.pi * x2)

    def grad_u(self, x: np.ndarray) -> np.ndarray:
        """
        Gradient of u.

        ∂u/∂x = π*cos(π*x)*sin(π*y)
        ∂u/∂y = π*sin(π*x)*cos(π*y)
        """
        x1, x2 = x[:, 0:1], x[:, 1:2]
        grad = np.zeros_like(x)
        grad[:, 0:1] = np.pi * np.cos(np.pi * x1) * np.sin(np.pi * x2)  # d/dx
        grad[:, 1:2] = np.pi * np.sin(np.pi * x1) * np.cos(np.pi * x2)  # d/dy
        return grad

    def laplace_u(self, x: np.ndarray) -> np.ndarray:
        """
        Laplacian of u.

        Δu = -2*π²*sin(π*x)*sin(π*y)
        """
        x1, x2 = x[:, 0:1], x[:, 1:2]
        return -2.0 * np.pi**2 * np.sin(np.pi * x1) * np.sin(np.pi * x2)
