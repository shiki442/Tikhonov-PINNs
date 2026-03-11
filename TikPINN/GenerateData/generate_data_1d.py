import numpy as np

def generate_data_1d(q_dagger, u_dagger, grad_u_x, laplace_u, noise_level):
    """
    Generate 1D data for PINN training.

    Parameters:
    -----------
    q_dagger : callable
        Function for q parameter
    u_dagger : callable
        Function for solution u
    grad_u_x : callable
        Function for gradient of u with respect to x
    laplace_u : callable
        Function for Laplacian of u
    noise_level : float
        Level of noise to add to measurements

    Returns:
    --------
    data_mat : numpy.ndarray
        Data matrix containing [int_point, bdy_point, normal_vec, m_int, m_bdy, f_val, g_val, u_int_val, q_val]
    """
    np.random.seed(2468)
    n_samples = 50000

    # Interior points
    n_samples_int = n_samples
    xq = np.random.rand(n_samples_int, 1)
    int_point = xq

    # u_int: solution
    u_int_val = u_dagger(xq)

    # f: source density
    laplace_u_int_val = laplace_u(xq)
    q_val = q_dagger(xq)
    f_val = -laplace_u_int_val + q_val * u_int_val

    # Boundary points
    n_samples_boundary = n_samples // 2

    bdy_point = np.vstack([np.zeros((n_samples_boundary, 1)), np.ones((n_samples_boundary, 1))])
    normal_vec = np.vstack([-np.ones((n_samples_boundary, 1)), np.ones((n_samples_boundary, 1))])

    # u_bdy: solution
    u_bdy_val = u_dagger(bdy_point)

    # g_val: boundary flux
    u_grad = grad_u_x(bdy_point).reshape(-1, 1)
    g_val = np.sum(u_grad * normal_vec, axis=1, keepdims=True)

    # Output data - add noise
    scale = np.linalg.norm(u_int_val, np.inf)
    m_int = u_int_val + noise_level * scale * np.random.randn(n_samples, 1)
    m_bdy = u_bdy_val + noise_level * scale * np.random.randn(n_samples_boundary * 2, 1)

    # Data matrix
    data_mat = np.hstack([int_point, bdy_point, normal_vec, m_int, m_bdy, f_val, g_val, u_int_val, q_val])

    # Shuffle data
    rand_index = np.random.permutation(n_samples)
    data_mat = data_mat[rand_index, :]

    return data_mat


# Test function
if __name__ == "__main__":
    # Define test functions
    def q_dagger(x):
        return np.ones_like(x)

    def u_dagger(x):
        return np.sin(np.pi * x)

    def grad_u_x(x):
        return np.pi * np.cos(np.pi * x)

    def laplace_u(x):
        return -np.pi**2 * np.sin(np.pi * x)

    # Generate data
    noise_level = 0.01
    data = generate_data_1d(q_dagger, u_dagger, grad_u_x, laplace_u, noise_level)

    print("Data generation successful!")
    print(f"Data shape: {data.shape}")
    print(f"First few rows of data:")
    print(data[:5, :])

    # Check data integrity
    assert data.shape[0] == 50000, f"Expected 50000 samples, got {data.shape[0]}"
    assert data.shape[1] == 9, f"Expected 9 columns, got {data.shape[1]}"
    print("\nAll tests passed!")