"""
Example 04: 2D problem with continuous but non-smooth q (circular inclusion with linear transition).

q(x,y) has a linear transition layer near the circular boundary:
  - q = 2.0 inside the inner circle (r <= r_inner)
  - q transitions linearly from 5.0 to 1.0 in the annulus (r_inner < r < r_outer)
  - q = 1.0 outside the outer circle (r >= r_outer)

This makes q continuous but not C^1 (the derivative is discontinuous at the boundaries).

u(x,y) = 4*x*(1-x)*y*(1-y)
"""

import numpy as np
from .problem_base_nd import ProblemND


class Example04Problem(ProblemND):
    """
    2D inverse problem with discontinuous parameter (circular inclusion).

    Problem definition:
        q_dagger(x, y) = 2.0 if (x-0.4)^2 + (y-0.4)^2 <= 0.15^2, else 1.0
        u_dagger(x, y) = 4*x*(1-x)*y*(1-y)

    Derivatives:
        ∂u/∂x = 4*(1-2x)*y*(1-y)
        ∂u/∂y = 4*x*(1-x)*(1-2y)
        Δu = -8*y*(1-y) - 8*x*(1-x) = -8*[y*(1-y) + x*(1-x)]
    """

    @property
    def name(self) -> str:
        return "example04"

    @property
    def dim(self) -> int:
        return 2

    def q_dagger(self, x: np.ndarray) -> np.ndarray:
        """
        True parameter q(x,y) with circular inclusion.

        q(x,y) = 5.0 inside circle centered at (0.4, 0.4) with radius 0.15
        q(x,y) = 1.0 outside the circle
        """
        x1, x2 = x[:, 0:1], x[:, 1:2]
        r_squared = (x1 - 0.4) ** 2 + (x2 - 0.4) ** 2
        radius_squared = 0.15 ** 2
        q = np.where(r_squared <= radius_squared, 2.0, 1.0)
        return q

    def u_dagger(self, x: np.ndarray) -> np.ndarray:
        """True solution u(x,y) = 4*x*(1-x)*y*(1-y)."""
        x1, x2 = x[:, 0:1], x[:, 1:2]
        return 4.0 * x1 * (1 - x1) * x2 * (1 - x2)

    def grad_u(self, x: np.ndarray) -> np.ndarray:
        """
        Gradient of u.

        ∂u/∂x = 4*(1-2x)*y*(1-y)
        ∂u/∂y = 4*x*(1-x)*(1-2y)
        """
        x1, x2 = x[:, 0:1], x[:, 1:2]
        grad = np.zeros_like(x)
        grad[:, 0:1] = 4.0 * (1 - 2 * x1) * x2 * (1 - x2)  # d/dx
        grad[:, 1:2] = 4.0 * x1 * (1 - x1) * (1 - 2 * x2)  # d/dy
        return grad

    def laplace_u(self, x: np.ndarray) -> np.ndarray:
        """
        Laplacian of u.

        Δu = -8*y*(1-y) - 8*x*(1-x) = -8*[y*(1-y) + x*(1-x)]
        """
        x1, x2 = x[:, 0:1], x[:, 1:2]
        return -8.0 * (x2 * (1 - x2) + x1 * (1 - x1))
