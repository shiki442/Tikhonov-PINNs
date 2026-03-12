"""
Problem base class for data generation.

This module provides an abstract base class for defining 1D inverse problems,
encapsulating the q, u functions and their derivatives for modular usage.
"""

from abc import ABC, abstractmethod
import numpy as np


class Problem1D(ABC):
    """
    Abstract base class for 1D inverse problems.

    This class encapsulates the problem definition including:
    - q_dagger: The true parameter function q(x)
    - u_dagger: The true solution function u(x)
    - grad_u_x: Gradient/derivative of u
    - laplace_u: Laplacian/second derivative of u
    - source_f: Source term f = -Δu + q*u
    - boundary_flux: Boundary flux g = ∇u · n

    Example:
        >>> class MyProblem(Problem1D):
        ...     @property
        ...     def name(self) -> str:
        ...         return "my_problem"
        ...     def q_dagger(self, x):
        ...         return np.ones_like(x)
        ...     def u_dagger(self, x):
        ...         return np.sin(np.pi * x)
        ...     # ... implement other methods
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Return the problem name used for file naming.

        Returns:
            str: Problem identifier, e.g., "example06"
        """
        pass

    @abstractmethod
    def q_dagger(self, x: np.ndarray) -> np.ndarray:
        """
        True parameter function q(x).

        Args:
            x: Input coordinates, shape (n, 1)

        Returns:
            q values at x, shape (n, 1)
        """
        pass

    @abstractmethod
    def u_dagger(self, x: np.ndarray) -> np.ndarray:
        """
        True solution function u(x).

        Args:
            x: Input coordinates, shape (n, 1)

        Returns:
            u values at x, shape (n, 1)
        """
        pass

    @abstractmethod
    def grad_u_x(self, x: np.ndarray) -> np.ndarray:
        """
        Gradient (first derivative) of u with respect to x.

        Args:
            x: Input coordinates, shape (n, 1)

        Returns:
            Gradient values at x, shape (n, 1)
        """
        pass

    @abstractmethod
    def laplace_u(self, x: np.ndarray) -> np.ndarray:
        """
        Laplacian (second derivative) of u.

        Args:
            x: Input coordinates, shape (n, 1)

        Returns:
            Laplacian values at x, shape (n, 1)
        """
        pass

    def source_f(self, x: np.ndarray) -> np.ndarray:
        """
        Compute source term f = -Δu + q*u.

        This is the default implementation using the abstract methods.
        Subclasses can override for custom behavior.

        Args:
            x: Input coordinates, shape (n, 1)

        Returns:
            Source term values, shape (n, 1)
        """
        return -self.laplace_u(x) + self.q_dagger(x) * self.u_dagger(x)

    def boundary_flux(self, x: np.ndarray, normal: np.ndarray) -> np.ndarray:
        """
        Compute boundary flux g = ∇u · n.

        Args:
            x: Boundary point coordinates, shape (n, 1)
            normal: Normal vectors at boundary points, shape (n, 1)

        Returns:
            Boundary flux values, shape (n, 1)
        """
        grad_u = self.grad_u_x(x).reshape(-1, 1)
        return np.sum(grad_u * normal, axis=1, keepdims=True)

    def evaluate_all(self, x: np.ndarray) -> dict:
        """
        Evaluate all functions at given points.

        Args:
            x: Input coordinates, shape (n, 1)

        Returns:
            Dictionary containing all function evaluations:
            - 'q_dagger': q(x)
            - 'u_dagger': u(x)
            - 'grad_u_x': ∇u(x)
            - 'laplace_u': Δu(x)
            - 'source_f': f(x) = -Δu + q*u
        """
        return {
            'q_dagger': self.q_dagger(x),
            'u_dagger': self.u_dagger(x),
            'grad_u_x': self.grad_u_x(x),
            'laplace_u': self.laplace_u(x),
            'source_f': self.source_f(x),
        }
