"""
Example Gaussian: 2D problem with Gaussian parameter peak.

q(x,y) = 1.0 + 2.0 * exp(-2 * ((x-0.6)^2 + (y-0.4)^2))
u(x,y) = 1.0 + 2.0 * sin(π*x) * sin(π*y)
"""

import numpy as np
from .problem_base_nd import ProblemND


class Example03Problem(ProblemND):
    """
    2D inverse problem with Gaussian parameter peak.

    Problem definition:
        q_dagger(x, y) = 1.0 + 2.0 * exp(-2 * ((x-0.6)^2 + (y-0.4)^2))
        u_dagger(x, y) = 1.0 + 2.0 * sin(π*x) * sin(π*y)

    Derivatives:
        ∂u/∂x = 2π*cos(π*x)*sin(π*y)
        ∂u/∂y = 2π*sin(π*x)*cos(π*y)
        Δu = -4*π²*sin(π*x)*sin(π*y)
    """

    @property
    def name(self) -> str:
        return "example_gaussian"

    @property
    def dim(self) -> int:
        return 2

    def q_dagger(self, x: np.ndarray) -> np.ndarray:
        """True parameter q(x,y) = 1.0 + 2.0 * exp(-5 * ((x-0.8)^2 + (y-0.2)^2))."""
        x1, x2 = x[:, 0:1], x[:, 1:2]
        r_squared = (x1 - 0.8) ** 2 + (x2 - 0.2) ** 2
        return 1.0 + 2.0 * np.exp(-5.0 * r_squared)

    def u_dagger(self, x: np.ndarray) -> np.ndarray:
        """True solution u(x,y) = 1.0 + 2.0 * sin(π*x) * sin(π*y)."""
        x1, x2 = x[:, 0:1], x[:, 1:2]
        return 1.0 + 2.0 * np.sin(np.pi * x1) * np.sin(np.pi * x2)

    def grad_u(self, x: np.ndarray) -> np.ndarray:
        """
        Gradient of u.

        ∂u/∂x = 2π*cos(π*x)*sin(π*y)
        ∂u/∂y = 2π*sin(π*x)*cos(π*y)
        """
        x1, x2 = x[:, 0:1], x[:, 1:2]
        grad = np.zeros_like(x)
        grad[:, 0:1] = 2.0 * np.pi * np.cos(np.pi * x1) * np.sin(np.pi * x2)  # d/dx
        grad[:, 1:2] = 2.0 * np.pi * np.sin(np.pi * x1) * np.cos(np.pi * x2)  # d/dy
        return grad

    def laplace_u(self, x: np.ndarray) -> np.ndarray:
        """
        Laplacian of u.

        Δu = -4*π²*sin(π*x)*sin(π*y)
        """
        x1, x2 = x[:, 0:1], x[:, 1:2]
        return -4.0 * np.pi**2 * np.sin(np.pi * x1) * np.sin(np.pi * x2)
