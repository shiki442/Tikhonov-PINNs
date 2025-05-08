import numpy as np
import os
from dolfin import *
from dolfin_adjoint import *

def generate_uniform_mesh(n):
    grid = np.linspace(0, 1, n+1)

    X, Y = np.meshgrid(grid, grid, indexing='ij')
    X, Y = X.reshape((-1, 1)), Y.reshape((-1, 1))
    xy = np.hstack((X, Y))
    return xy

def params2str(lamb, noise_level):
    # Convert parameters to string
    lamb_str = int(-np.log10(lamb)) if lamb != 0 else 0
    noise_level_str = int(100 * noise_level)
    str = f"{lamb_str:02d}_{noise_level_str:02d}"
    return str

def str2params(str):
    # Convert string parameters to float
    lamb_str, noise_level_str = str.split('_')
    lamb = 10 ** (-float(lamb_str))
    noise_level = float(noise_level_str) / 100
    return lamb, noise_level

def save_result(u, q, lamb, noise_level, path):
    # Get value of the state and potential
    d = 1.0
    p0 = Point(0.0, 0.0)
    p1 = Point(d, d)
    mesh = RectangleMesh(p0, p1, 500, 500)
    V_fine = FunctionSpace(mesh, "Lagrange", 1)
    u_fine = interpolate(u, V_fine)
    q_fine = interpolate(q, V_fine)
    u_val = u_fine.compute_vertex_values()
    q_val = q_fine.compute_vertex_values()
    xy = generate_uniform_mesh(500)
    x = np.hstack((xy, q_val[:,None], u_val[:,None]))
    # Save the result to a file
    str_params = params2str(lamb, noise_level)
    path_out = os.path.join(path, f"result_{str_params}/result.txt")
    os.makedirs(os.path.dirname(path_out), exist_ok=True)
    np.savetxt(path_out, x, delimiter=",")
    print(f"Results saved to {path_out}")
    return

def save_err_q(err_q_history, lamb, noise_level, path):
    str_params = params2str(lamb, noise_level)
    path = os.path.join(path, f"result_{str_params}/err_with_iter.txt")
    print(f"Data saved to: {path}")
    np.savetxt(path, err_q_history, delimiter=",")
    return

x = generate_uniform_mesh(50)