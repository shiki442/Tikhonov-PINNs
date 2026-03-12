"""
Example nD problem: Separable sine product.

Generalizes the 1D sine problem to n dimensions:
    q(x) = constant
    u(x) = ∏ᵢ sin(π * xᵢ)
"""

import numpy as np
from .problem_base_nd import ProblemND


class SineProductProblem(ProblemND):
    """
    nD inverse problem with separable sine product solution.

    Problem definition (for dim dimensions):
        q_dagger(x) = dim * π²
        u_dagger(x) = ∏ᵢ₌₁ᵈⁱᵐ sin(π * xᵢ)

    Derivatives:
        ∂u/∂xᵢ = π * cos(π * xᵢ) * ∏ⱼ≠ᵢ sin(π * xⱼ)
        Δu = -dim * π² * ∏ᵢ sin(π * xᵢ)
    """

    def __init__(self, dim: int = 2):
        """
        Initialize the problem with specified dimension.

        Args:
            dim: Problem dimension (default: 2)
        """
        self._dim = dim

    @property
    def name(self) -> str:
        """Return problem name for file naming."""
        return f"sine_product_{self._dim}d"

    @property
    def dim(self) -> int:
        """Return problem dimension."""
        return self._dim

    def q_dagger(self, x: np.ndarray) -> np.ndarray:
        """
        True parameter q(x) = dim * π².

        This constant ensures that -Δu + q*u = 0 for the given u.
        """
        return np.full((x.shape[0], 1), self._dim * np.pi**2)

    def u_dagger(self, x: np.ndarray) -> np.ndarray:
        """True solution u(x) = ∏ᵢ sin(π * xᵢ)."""
        result = np.ones((x.shape[0], 1))
        for i in range(self._dim):
            result *= np.sin(np.pi * x[:, i:i+1])
        return result

    def grad_u(self, x: np.ndarray) -> np.ndarray:
        """
        Gradient of u.

        ∂u/∂xᵢ = π * cos(π * xᵢ) * ∏ⱼ≠ᵢ sin(π * xⱼ)
        """
        grad = np.zeros_like(x)
        for i in range(self._dim):
            # Product of all sin terms except i
            product = np.ones((x.shape[0], 1))
            for j in range(self._dim):
                if j != i:
                    product *= np.sin(np.pi * x[:, j:j+1])
            grad[:, i:i+1] = np.pi * np.cos(np.pi * x[:, i:i+1]) * product
        return grad

    def laplace_u(self, x: np.ndarray) -> np.ndarray:
        """
        Laplacian of u.

        Δu = -dim * π² * ∏ᵢ sin(π * xᵢ) = -dim * π² * u
        """
        u = self.u_dagger(x)
        return -self._dim * np.pi**2 * u
