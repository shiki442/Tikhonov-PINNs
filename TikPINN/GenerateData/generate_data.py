"""
Data generation module for PINN training.

Supports both 1D and nD problems with unified interface.
"""

import numpy as np
import os
import torch

from problems.problem_base_1d import Problem1D
from problems.problem_base_nd import ProblemND


# =============================================================================
# Boundary generation (nD)
# =============================================================================

def generate_boundary_points_nd(dim: int, n_samples_per_face: int) -> tuple:
    """
    Generate boundary points on a hypercube [0, 1]^dim.

    For a dim-dimensional hypercube, there are 2*dim faces.
    Each face is a (dim-1)-dimensional hypercube.

    Parameters:
    -----------
    dim : int
        Dimension of the domain
    n_samples_per_face : int
        Number of samples on each face

    Returns:
    --------
    tuple: (bdy_points, normal_vec)
        bdy_points : np.ndarray, shape (n_bdy, dim)
            Boundary point coordinates
        normal_vec : np.ndarray, shape (n_bdy, dim)
            Outward normal vectors at boundary points
    """
    bdy_points_list = []
    normal_vec_list = []

    # For each dimension, we have two faces: x_i = 0 and x_i = 1
    for d in range(dim):
        # Face at x_d = 0 (normal points in -e_d direction)
        points = np.random.rand(n_samples_per_face, dim)
        points[:, d] = 0

        normals = np.zeros((n_samples_per_face, dim))
        normals[:, d] = -1

        bdy_points_list.append(points)
        normal_vec_list.append(normals)

        # Face at x_d = 1 (normal points in +e_d direction)
        points = np.random.rand(n_samples_per_face, dim)
        points[:, d] = 1

        normals = np.zeros((n_samples_per_face, dim))
        normals[:, d] = 1

        bdy_points_list.append(points)
        normal_vec_list.append(normals)

    bdy_points = np.vstack(bdy_points_list)
    normal_vec = np.vstack(normal_vec_list)

    return bdy_points, normal_vec


# =============================================================================
# Data generation (1D)
# =============================================================================

def generate_data_1d(problem: Problem1D, noise_level: float, n_samples: int = 50000) -> dict:
    """
    Generate 1D data for PINN training.

    Parameters:
    -----------
    problem : Problem1D
        Problem instance containing q_dagger, u_dagger, grad_u_x, laplace_u methods
    noise_level : float
        Level of noise to add to measurements
    n_samples : int, optional
        Number of interior sample points (default: 50000)

    Returns:
    --------
    dict: Data dictionary containing:
        - 'int_points': Interior point coordinates, shape (n_samples, 1)
        - 'bdy_points': Boundary point coordinates, shape (n_bdy, 1)
        - 'normal_vec': Normal vectors at boundary, shape (n_bdy, 1)
        - 'm_int': Noisy interior measurements, shape (n_samples, 1)
        - 'm_bdy': Noisy boundary measurements, shape (n_bdy, 1)
        - 'f_val': Source term values, shape (n_samples, 1)
        - 'g_val': Boundary flux values, shape (n_bdy, 1)
        - 'u_dagger': True solution at interior points, shape (n_samples, 1)
        - 'q_dagger': True parameter at interior points, shape (n_samples, 1)
        - 'noise_level': Noise level used (float)
        - 'problem_name': Problem name (str)
        - 'n_samples_int': Number of interior samples (int)
        - 'n_samples_bdy': Number of boundary samples (int)
    """
    np.random.seed(2468)

    # Interior points
    n_samples_int = n_samples
    xq = np.random.rand(n_samples_int, 1)
    int_points = xq

    # Evaluate problem functions at interior points
    u_int_val = problem.u_dagger(xq)
    q_val = problem.q_dagger(xq)
    f_val = problem.source_f(xq)

    # Boundary points
    n_samples_boundary = n_samples // 2
    bdy_points = np.vstack([np.zeros((n_samples_boundary, 1)),
                             np.ones((n_samples_boundary, 1))])
    normal_vec = np.vstack([-np.ones((n_samples_boundary, 1)),
                             np.ones((n_samples_boundary, 1))])

    # Evaluate at boundary points
    u_bdy_val = problem.u_dagger(bdy_points)
    g_val = problem.boundary_flux(bdy_points, normal_vec)

    # Add noise to measurements
    scale = np.linalg.norm(u_int_val, np.inf)
    m_int = u_int_val + noise_level * scale * np.random.randn(n_samples, 1)
    m_bdy = u_bdy_val + noise_level * scale * np.random.randn(n_samples_boundary * 2, 1)

    # Build data dictionary
    data = {
        'int_points': int_points,
        'bdy_points': bdy_points,
        'normal_vec': normal_vec,
        'm_int': m_int,
        'm_bdy': m_bdy,
        'f_val': f_val,
        'g_val': g_val,
        'u_dagger': u_int_val,
        'q_dagger': q_val,
        'noise_level': noise_level,
        'problem_name': problem.name,
        'n_samples_int': n_samples_int,
        'n_samples_bdy': n_samples_boundary * 2,
    }

    return data


# =============================================================================
# Data generation (nD)
# =============================================================================

def generate_data_nd(problem: ProblemND, noise_level: float,
                     n_samples_int: int = 50000,
                     n_samples_per_face: int = 5000) -> dict:
    """
    Generate nD data for PINN training.

    Parameters:
    -----------
    problem : ProblemND
        Problem instance containing q_dagger, u_dagger, grad_u, laplace_u methods
    noise_level : float
        Level of noise to add to measurements
    n_samples_int : int, optional
        Number of interior sample points (default: 50000)
    n_samples_per_face : int, optional
        Number of samples per boundary face (default: 5000)
        Total boundary samples = 2 * dim * n_samples_per_face

    Returns:
    --------
    dict: Data dictionary containing:
        - 'int_points': Interior point coordinates, shape (n_samples_int, dim)
        - 'bdy_points': Boundary point coordinates, shape (n_bdy, dim)
        - 'normal_vec': Normal vectors at boundary, shape (n_bdy, dim)
        - 'm_int': Noisy interior measurements, shape (n_samples_int, 1)
        - 'm_bdy': Noisy boundary measurements, shape (n_bdy, 1)
        - 'f_val': Source term values, shape (n_samples_int, 1)
        - 'g_val': Boundary flux values, shape (n_bdy, 1)
        - 'u_dagger': True solution at interior points, shape (n_samples_int, 1)
        - 'q_dagger': True parameter at interior points, shape (n_samples_int, 1)
        - 'noise_level': Noise level used (float)
        - 'problem_name': Problem name (str)
        - 'dim': Problem dimension (int)
        - 'n_samples_int': Number of interior samples (int)
        - 'n_samples_bdy': Number of boundary samples (int)
    """
    np.random.seed(2468)

    dim = problem.dim

    # Interior points - uniformly sampled from [0, 1]^dim
    int_points = np.random.rand(n_samples_int, dim)

    # Evaluate problem functions at interior points
    u_int_val = problem.u_dagger(int_points)
    q_val = problem.q_dagger(int_points)
    f_val = problem.source_f(int_points)

    # Boundary points - sampled from faces of hypercube
    bdy_points, normal_vec = generate_boundary_points_nd(dim, n_samples_per_face)
    n_bdy = bdy_points.shape[0]

    # Evaluate at boundary points
    u_bdy_val = problem.u_dagger(bdy_points)
    g_val = problem.boundary_flux(bdy_points, normal_vec)

    # Add noise to measurements
    scale = np.linalg.norm(u_int_val, np.inf)
    m_int = u_int_val + noise_level * scale * np.random.randn(n_samples_int, 1)
    m_bdy = u_bdy_val + noise_level * scale * np.random.randn(n_bdy, 1)

    # Build data dictionary
    data = {
        'int_points': int_points,
        'bdy_points': bdy_points,
        'normal_vec': normal_vec,
        'm_int': m_int,
        'm_bdy': m_bdy,
        'f_val': f_val,
        'g_val': g_val,
        'u_dagger': u_int_val,
        'q_dagger': q_val,
        'noise_level': noise_level,
        'problem_name': problem.name,
        'dim': dim,
        'n_samples_int': n_samples_int,
        'n_samples_bdy': n_bdy,
    }

    return data


# =============================================================================
# Save/Load utilities
# =============================================================================

def save_data_pt(data: dict, file_path: str):
    """
    Save data dictionary to PyTorch .pt file.

    Parameters:
    -----------
    data : dict
        Data dictionary containing numpy arrays and metadata
    file_path : str
        Path to save the .pt file
    """
    # Convert numpy arrays to torch tensors
    data_tensor = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            data_tensor[key] = torch.tensor(value, dtype=torch.float32)
        else:
            data_tensor[key] = value

    # Ensure directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Save with torch.save
    torch.save(data_tensor, file_path)


def load_data_pt(file_path: str) -> dict:
    """
    Load data from PyTorch .pt file.

    Parameters:
    -----------
    file_path : str
        Path to the .pt file

    Returns:
    --------
    dict: Data dictionary containing tensors and metadata
    """
    data = torch.load(file_path, weights_only=False)
    return data


# =============================================================================
# Legacy functions for backward compatibility
# =============================================================================

def generate_data_1d_legacy(q_dagger, u_dagger, grad_u_x, laplace_u, noise_level):
    """
    Legacy function for backward compatibility.
    Use generate_data_1d with a Problem1D instance instead.
    """
    class LegacyProblem(Problem1D):
        @property
        def name(self):
            return "legacy"

        def q_dagger(self, x):
            return q_dagger(x)

        def u_dagger(self, x):
            return u_dagger(x)

        def grad_u_x(self, x):
            return grad_u_x(x)

        def laplace_u(self, x):
            return laplace_u(x)

    problem = LegacyProblem()
    return generate_data_1d(problem, noise_level)


# =============================================================================
# Test functions
# =============================================================================

if __name__ == "__main__":
    # Test 1D
    print("=" * 60)
    print("Testing 1D data generation")
    print("=" * 60)

    class TestProblem1D(Problem1D):
        @property
        def name(self):
            return "test_problem_1d"

        def q_dagger(self, x):
            return np.ones_like(x)

        def u_dagger(self, x):
            return np.sin(np.pi * x)

        def grad_u_x(self, x):
            return np.pi * np.cos(np.pi * x)

        def laplace_u(self, x):
            return -np.pi**2 * np.sin(np.pi * x)

    problem_1d = TestProblem1D()
    data_1d = generate_data_1d(problem_1d, 0.01)

    print(f"Problem name: {data_1d['problem_name']}")
    print(f"Noise level: {data_1d['noise_level']}")
    print(f"Interior samples: {data_1d['n_samples_int']}")
    print(f"Boundary samples: {data_1d['n_samples_bdy']}")
    print("\nData keys:", list(data_1d.keys()))
    for key, value in data_1d.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: shape {value.shape}, dtype {value.dtype}")

    print("\n" + "=" * 60)
    print("Testing nD data generation")
    print("=" * 60)

    # Test 2D
    class TestProblem2D(ProblemND):
        @property
        def name(self):
            return "test_problem_2d"

        @property
        def dim(self):
            return 2

        def q_dagger(self, x):
            return np.ones((x.shape[0], 1))

        def u_dagger(self, x):
            return np.sin(np.pi * x[:, 0:1]) * np.sin(np.pi * x[:, 1:2])

        def grad_u(self, x):
            grad = np.zeros_like(x)
            grad[:, 0:1] = np.pi * np.cos(np.pi * x[:, 0:1]) * np.sin(np.pi * x[:, 1:2])
            grad[:, 1:2] = np.pi * np.sin(np.pi * x[:, 0:1]) * np.cos(np.pi * x[:, 1:2])
            return grad

        def laplace_u(self, x):
            return -2 * np.pi**2 * np.sin(np.pi * x[:, 0:1]) * np.sin(np.pi * x[:, 1:2])

    problem_2d = TestProblem2D()
    data_2d = generate_data_nd(problem_2d, 0.01, n_samples_int=10000, n_samples_per_face=1000)

    print(f"Problem name: {data_2d['problem_name']}")
    print(f"Dimension: {data_2d['dim']}")
    print(f"Interior samples: {data_2d['n_samples_int']}")
    print(f"Boundary samples: {data_2d['n_samples_bdy']}")

    # Test save/load
    test_file = "./data/test_data.pt"
    save_data_pt(data_2d, test_file)
    print(f"\nSaved to {test_file}")

    loaded_data = load_data_pt(test_file)
    print("Loaded data successfully!")

    print("\nAll tests passed!")
