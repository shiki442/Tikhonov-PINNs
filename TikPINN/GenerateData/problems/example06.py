"""
Example 06 problem definition.

q(x) = (1 + x) * exp(x)
u(x) = 1 + sin(π * x)
"""

import numpy as np
from .problem_base_1d import Problem1D


class Example06Problem(Problem1D):
    """
    Example 06: 1D inverse problem with exponential q and sinusoidal u.

    Problem definition:
        q_dagger(x) = (1 + x) * exp(x)
        u_dagger(x) = 1 + sin(π * x)

    Derivatives:
        grad_u_x(x) = π * cos(π * x)
        laplace_u(x) = -π^2 * sin(π * x)
    """

    @property
    def name(self) -> str:
        """Return problem name for file naming."""
        return "example06"

    def q_dagger(self, x: np.ndarray) -> np.ndarray:
        """True parameter q(x) = (1 + x) * exp(x)."""
        return (1 + x) * np.exp(x)

    def u_dagger(self, x: np.ndarray) -> np.ndarray:
        """True solution u(x) = 1 + sin(π * x)."""
        return 1.0 + np.sin(np.pi * x)

    def grad_u_x(self, x: np.ndarray) -> np.ndarray:
        """First derivative: u'(x) = π * cos(π * x)."""
        return np.pi * np.cos(np.pi * x)

    def laplace_u(self, x: np.ndarray) -> np.ndarray:
        """Second derivative: u''(x) = -π^2 * sin(π * x)."""
        return -np.pi**2 * np.sin(np.pi * x)
