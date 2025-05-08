import numpy as np
import os

def generate_date(q_dagger, u_dagger, grad_u_x, grad_u_y, laplace_u, noise_level):
    np.random.seed(2468)
    n_samples = 100000
    n_obs = 50

    # Interior points
    n_samples_int = n_samples
    xq = np.random.rand(n_samples_int, 1)
    yq = np.random.rand(n_samples_int, 1)
    int_point = np.hstack((xq, yq))
    
    # Calculate interior values
    u_int_val = u_dagger(xq, yq)
    laplace_u_int_val = laplace_u(xq, yq)
    q_val = q_dagger(xq, yq)
    f_val = -laplace_u_int_val + q_val * u_int_val

    # Boundary points
    n_samples_boundary = n_samples // 4
    left = np.hstack((np.zeros((n_samples_boundary, 1)), np.random.rand(n_samples_boundary, 1)))
    right = np.hstack((np.ones((n_samples_boundary, 1)), np.random.rand(n_samples_boundary, 1)))
    lower = np.hstack((np.random.rand(n_samples_boundary, 1), np.zeros((n_samples_boundary, 1))))
    upper = np.hstack((np.random.rand(n_samples_boundary, 1), np.ones((n_samples_boundary, 1))))
    bdy_point = np.vstack((left, right, lower, upper))
    
    # Normal vectors
    normal_vec_left = np.tile([-1, 0], (n_samples_boundary, 1))
    normal_vec_right = np.tile([1, 0], (n_samples_boundary, 1))
    normal_vec_lower = np.tile([0, -1], (n_samples_boundary, 1))
    normal_vec_upper = np.tile([0, 1], (n_samples_boundary, 1))
    normal_vec = np.vstack((normal_vec_left, normal_vec_right, normal_vec_lower, normal_vec_upper))

    # Calculate boundary values
    u_bdy_val = u_dagger(bdy_point[:, 0:1], bdy_point[:, 1:2])
    grad_u_x_val = grad_u_x(bdy_point[:, 0:1], bdy_point[:, 1:2])
    grad_u_y_val = grad_u_y(bdy_point[:, 0:1], bdy_point[:, 1:2])
    u_grad = np.hstack((grad_u_x_val, grad_u_y_val))
    g_val = np.sum(u_grad * normal_vec, axis=1, keepdims=True)

    # Create observation points
    x = np.linspace(0, 1, n_obs)
    y = np.linspace(0, 1, n_obs)
    xx, yy = np.meshgrid(x, y)
    int_obs_point = np.vstack([xx.ravel(), yy.ravel()]).T
    u_int_obs = u_dagger(int_obs_point[:, 0:1], int_obs_point[:, 1:2])

    # Create boundary observation points
    left = np.column_stack((np.zeros(n_obs * n_obs // 4), np.linspace(0, 1, n_obs * n_obs // 4)))
    right = np.column_stack((np.ones(n_obs * n_obs // 4), np.linspace(0, 1, n_obs * n_obs // 4)))
    bottom = np.column_stack((np.linspace(0, 1, n_obs * n_obs // 4), np.zeros(n_obs * n_obs // 4)))
    top = np.column_stack((np.linspace(0, 1, n_obs * n_obs // 4), np.ones(n_obs * n_obs // 4)))
    bdy_obs_point = np.vstack((left, right, bottom, top))
    u_bdy_obs = u_dagger(bdy_obs_point[:, 0:1], bdy_obs_point[:, 1:2])

    # Add noise
    scale = np.linalg.norm(u_int_obs, ord=np.inf)
    m_int = u_int_obs + noise_level * scale * np.random.randn(n_obs * n_obs, 1)
    scale = np.linalg.norm(u_bdy_obs, ord=np.inf)
    m_bdy = u_bdy_obs + noise_level * scale * np.random.randn(n_obs * n_obs, 1)

    # Create data matrix
    data_mat = np.hstack((
        int_point,
        bdy_point,
        normal_vec,
        f_val,
        g_val,
        u_int_val,
        q_val
    ))

    obs_mat = np.hstack((
        int_obs_point,
        bdy_obs_point,
        m_int,
        m_bdy,
    ))

    # Shuffle rows
    data_mat = data_mat[np.random.permutation(n_samples), :]
    obs_mat = obs_mat[np.random.permutation(n_obs * n_obs), :]
    
    return data_mat, obs_mat

if __name__ == "__main__":
    # Define functions for example01
    q_dagger = lambda x, y: 1.0 + 0.5 * np.sin(np.pi * x) * np.sin(np.pi * y)
    u_dagger = lambda x, y: 1.0 + np.sin(np.pi * x) * np.sin(np.pi * y)
    grad_u_x = lambda x, y: np.pi * np.cos(np.pi * x) * np.sin(np.pi * y)
    grad_u_y = lambda x, y: np.pi * np.sin(np.pi * x) * np.cos(np.pi * y)
    laplace_u = lambda x, y: -2.0 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)

    # Create data directory if not exists
    os.makedirs('./data', exist_ok=True)

    # Generate data for different noise levels
    deltas = [0.01, 0.10, 0.20, 0.50]
    for delta in deltas:
        data_mat, obs_mat = generate_date(q_dagger, u_dagger, grad_u_x, grad_u_y, laplace_u, delta)

        file_name = os.path.join('data', f'example01data{int(100*delta):02d}.txt')
        abs_path = os.path.abspath(file_name)
        print(f"[DEBUG] 文件将保存到：{abs_path}")
        print(f"[DEBUG] 数据矩阵维度：{data_mat.shape}")
        np.savetxt(file_name, data_mat, delimiter=',')
        print(f'{file_name} finished.')

        obs_file_name = os.path.join('data', f'obs01data{int(100*delta):02d}.txt')
        abs_obs_path = os.path.abspath(obs_file_name)
        print(f"[DEBUG] 观测值将保存到：{abs_obs_path}")
        print(f"[DEBUG] 观测数据矩阵维度：{obs_mat.shape}")
        np.savetxt(obs_file_name, obs_mat, delimiter=',')
        print(f'{obs_file_name} finished.')