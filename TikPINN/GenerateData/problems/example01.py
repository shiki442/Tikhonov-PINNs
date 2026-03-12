"""
Example 01: 2D sine product problem.

q(x,y) = 1.0 + 0.5*sin(π*x)*sin(π*y)
u(x,y) = 1.0 + sin(π*x)*sin(π*y)
"""

import numpy as np
from .problem_base_nd import ProblemND


class Example01Problem(ProblemND):
    """
    Example 01: 2D inverse problem with sine product solution.

    Problem definition:
        q_dagger(x, y) = 1.0 + 0.5*sin(π*x)*sin(π*y)
        u_dagger(x, y) = 1.0 + sin(π*x)*sin(π*y)

    Derivatives:
        ∂u/∂x = π*cos(π*x)*sin(π*y)
        ∂u/∂y = π*sin(π*x)*cos(π*y)
        Δu = -2*π²*sin(π*x)*sin(π*y)
    """

    @property
    def name(self) -> str:
        return "example01"

    @property
    def dim(self) -> int:
        return 2

    def q_dagger(self, x: np.ndarray) -> np.ndarray:
        """True parameter q(x,y) = 1.0 + 0.5*sin(π*x)*sin(π*y)."""
        x1, x2 = x[:, 0:1], x[:, 1:2]
        return 1.0 + 0.5 * np.sin(np.pi * x1) * np.sin(np.pi * x2)

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
