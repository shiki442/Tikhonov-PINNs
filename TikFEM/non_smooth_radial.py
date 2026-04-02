from dolfin import *
from dolfin_adjoint import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm  # For custom color mapping
import random
from utils import save_result, save_err_q

np.random.seed(123)
random.seed(123)
parameters["reorder_dofs_serial"] = False
# set_log_level(LogLevel.ERROR)

# Create computational domain
d = 1.0
p0 = Point(0.0, 0.0)
p1 = Point(d, d)
mesh = RectangleMesh(p0, p1, 320, 320)

# Create `MeshFunction` to mark boundaries
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)


class Left(SubDomain):  # Left boundary x = 0
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0)


class Right(SubDomain):  # Right boundary x = d
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], d)


class Top(SubDomain):  # Top boundary y = d
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], d)


class Bottom(SubDomain):  # Bottom boundary y = 0
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 0)


def Inverse_potential(noise_level, lamb, max_iter=30):
    # Define finite element space
    V = FunctionSpace(mesh, "Lagrange", 2)
    W = FunctionSpace(mesh, "Lagrange", 1)

    # Define trial and test functions
    u = Function(V, name='State')
    v = TestFunction(V)
    # Better initial guess: Gaussian peak centered at (0.4, 0.4)
    q = interpolate(Expression("1.0 + 4.0*exp(-4*((x[0]-0.4)*(x[0]-0.4) + (x[1]-0.4)*(x[1]-0.4)))", degree=2), V)

    # Calculate the analytic f
    class RegionExpression(UserExpression):
        def eval(self, value, x):
            r = np.sqrt((x[0] - 0.4) ** 2 + (x[1] - 0.4) ** 2)
            r_inner = 0.15
            r_outer = 0.2
            if r <= r_inner:
                value[0] = 5.0
            elif r < r_outer:
                value[0] = 5.0 + (1.0 - 5.0) * (r - r_inner) / (r_outer - r_inner)
            else:
                value[0] = 1.0

        def value_shape(self):
            return ()

    alpha = Constant(1.0)
    u_ex = Expression("4*x[0]*(1-x[0])*x[1]*(1-x[1])", degree=1)
    q_ex = RegionExpression(degree=1)

    # f = -Delta(u_ex) + u_ex * q_ex
    # u_ex = 4*x*(1-x)*y*(1-y)
    # d2u_ex/dx2 = -8*y*(1-y)
    # d2u_ex/dy2 = -8*x*(1-x)
    D2u_ex = Expression("-8*x[1]*(1-x[1]) - 8*x[0]*(1-x[0])", degree=1)
    f = -D2u_ex + u_ex * q_ex

    # Generate the observation data for the state variable
    u_obs = interpolate(u_ex, V)
    scale = u_obs.vector().max()
    random_noise = scale * noise_level * np.random.randn(len(u.vector()))
    u_obs.vector()[:] += random_noise

    # Assign different numbers to different boundaries
    Left().mark(boundaries, 1)
    Right().mark(boundaries, 2)
    Top().mark(boundaries, 3)
    Bottom().mark(boundaries, 4)

    # Create marked measure ds
    ds = Measure("ds", domain=mesh, subdomain_data=boundaries)

    # Define different Neumann conditions
    # g = du_ex/dn = grad(u_ex) · n
    # grad(u_ex) = [4*(1-2x)*y*(1-y), 4*x*(1-x)*(1-2y)]
    g1 = Expression("-4*(1-2*x[0])*x[1]*(1-x[1])", degree=1)  # Left boundary (n = [-1, 0])
    g2 = Expression("4*(1-2*x[0])*x[1]*(1-x[1])", degree=1)  # Right boundary (n = [1, 0])
    g3 = Expression("4*x[0]*(1-x[0])*(1-2*x[1])", degree=1)  # Top boundary (n = [0, 1])
    g4 = Expression("-4*x[0]*(1-x[0])*(1-2*x[1])", degree=1)  # Bottom boundary (n = [0, -1])

    # Define the variational form
    a = alpha * inner(grad(u), grad(v)) * dx + q * u * v * dx
    L = f * v * dx + g1 * v * ds(1) + g2 * v * ds(2) + g3 * v * ds(3) + g4 * v * ds(4)
    F = a - L
    solve(F == 0, u)

    # Define the cost functional and Tikhonov regularization term
    cost = 0.5 * (u - u_obs) ** 2 * dx
    hess_q = grad(grad(q))
    # Tikh_reg = (q ** 2) * dx
    Tikh_reg = (q**2 + inner(grad(q), grad(q)) + inner(hess_q, hess_q)) * dx

    err_q_history = []
    state_history = []

    # Define the callback function to store control history
    def eval_cb(control_value):
        control_error = errornorm(q_ex, control_value) / norm(interpolate(q_ex, V))
        err_q_history.append(control_error)

    # Define the optimization problem
    J = assemble(cost) + lamb * assemble(Tikh_reg)
    control = Control(q)
    rf = ReducedFunctional(J, control, eval_cb_pre=eval_cb)
    q_opt = minimize(rf, bounds=(1.0, 5.0), tol=1e-10, options={"gtol": 1e-10, "maxiter": max_iter})

    # Recompute the state variable with the optimal control
    q.assign(q_opt)
    solve(F == 0, u)

    # Compute the error
    state_error = errornorm(u_ex, u) / norm(interpolate(u_ex, V))
    control_error = errornorm(q_ex, q) / norm(interpolate(q_ex, V))
    print("h(min):           %e." % mesh.hmin())
    print("Error in state:   %e." % state_error)
    print("Error in control: %e." % control_error)

    save_result(u, q, lamb, noise_level, path="./output/non_smooth_radial/")
    save_err_q(err_q_history, lamb, noise_level, path="./output/non_smooth_radial/")

    tape = get_working_tape()
    tape.clear_tape()
    return J, state_error, err_q_history


if __name__ == "__main__":
    lamb_list = [1e-9]
    delta_list = [0.01]
    for lamb in lamb_list:
        for delta in delta_list:
            print(f"Noise level: {delta}, Lambda: {lamb}")
            _, _, err_q = Inverse_potential(delta, lamb, max_iter=200)
