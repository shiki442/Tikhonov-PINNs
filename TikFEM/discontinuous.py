from dolfin import *
from dolfin_adjoint import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm  # For custom color mapping
import random
from utils import generate_uniform_mesh, save_result

np.random.seed(123)
random.seed(123)
parameters["reorder_dofs_serial"] = False
set_log_level(LogLevel.ERROR) 

d = 1.0
# Create computational domain
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

def Inverse_potential(noise_level, lamb):
    # Define finite element space
    V = FunctionSpace(mesh, "Lagrange", 1)
    W = FunctionSpace(mesh, "DG", 1)

    # Define trial and test functions
    u = Function(V, name='State')
    v = TestFunction(V)
    q = interpolate(Constant(1.0), V)

    # Calculate the alynatic f
    class RegionExpression(UserExpression):
        def eval(self, value, x):
            r2 = (x[0]-0.5)**2 + (x[1]-0.5)**2
            if x[0] >= x[1]:
                value[0] = 1.0 + 0.5 * np.exp(-9 * r2)
            else:
                value[0] = 1.0

        def value_shape(self):
            return ()

    alpha = Constant(1.0)
    u_ex = Expression("1+sin(pi*x[0])*sin(pi*x[1])", degree=1)
    q_ex = RegionExpression(degree=1)
    D2u_ex = Expression("-2*pi*pi*sin(pi*x[0])*sin(pi*x[1])", degree=1)
    f = -D2u_ex + u_ex * q_ex

    # Generate the obesevation data for the state variable
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
    g1 = Expression("-pi*cos(pi*x[0])*sin(pi*x[1])", degree=1)  # Left boundary
    g2 = Expression("pi*cos(pi*x[0])*sin(pi*x[1])", degree=1)   # Right boundary
    g3 = Expression("pi*sin(pi*x[0])*cos(pi*x[1])", degree=1)   # Top boundary
    g4 = Expression("-pi*sin(pi*x[0])*cos(pi*x[1])", degree=1)  # Bottom boundary

    # Define the variational form
    a = alpha * inner(grad(u), grad(v)) * dx + q * u * v * dx
    L = f * v * dx + g1 * v * ds(1) + g2 * v * ds(2) + g3 * v * ds(3) + g4 * v * ds(4)
    F = a - L
    solve(F == 0, u)

    # Define the cost functional and Tikhonov regularization term
    cost = 0.5 * (u - u_obs)**2 * dx
    hess_q = grad(grad(q))
    Tikh_reg = (q**2 + inner(grad(q), grad(q)) + inner(hess_q, hess_q)) * dx

    # Define the optimization problem
    J = assemble(cost) + lamb * assemble(Tikh_reg)
    control = Control(q)
    rf = ReducedFunctional(J, control)
    q_opt = minimize(rf, bounds=(-3.0, 3.0), tol=1e-10, options={"gtol": 1e-10, "disp": True, "maxiter": 30})

    # Recompute the state variable with the optimal control
    q.assign(q_opt)
    solve(F == 0, u)

    # Compute the error
    state_error = errornorm(u_ex, u) / norm(interpolate(u_ex, V))
    control_error = errornorm(q_ex, q) / norm(interpolate(q_ex, V))
    print("h(min):           %e." % mesh.hmin())
    print("Error in state:   %e." % state_error)
    print("Error in control: %e." % control_error)

    save_result(u, q, lamb, noise_level, path="./discontinuous/")
    
    tape = get_working_tape()
    tape.clear_tape()
    return

if __name__ == "__main__":
    lamb_list = [1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 0.0]
    delta_list = [0.01, 0.1, 0.2, 0.5]
    for lamb in lamb_list:
        for delta in delta_list:
            print(f"Noise level: {delta}, Lambda: {lamb}")
            Inverse_potential(delta, lamb)