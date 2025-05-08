from dolfin import *
from dolfin_adjoint import *

d = 1.0
# 创建计算区域（单位正方形）
p0 = Point(0.0, 0.0)
p1 = Point(d, d)
mesh = RectangleMesh(p0, p1, 16, 16)

# 定义有限元空间
V = FunctionSpace(mesh, "CG", 1)

# 定义试探函数和测试函数
u = Function(V, name='State')
v = TestFunction(V)

# 创建 `MeshFunction` 来标记边界
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)

# 定义四条边的编号
class Left(SubDomain):  # 左边界 x = 0
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0)

class Right(SubDomain):  # 右边界 x = d
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], d)

class Top(SubDomain):  # 上边界 y = d
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], d)

class Bottom(SubDomain):  # 下边界 y = 0
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 0)

# 给不同边界赋不同的编号
Left().mark(boundaries, 1)
Right().mark(boundaries, 2)
Top().mark(boundaries, 3)
Bottom().mark(boundaries, 4)

# 创建带标记的测度 ds
ds = Measure("ds", domain=mesh, subdomain_data=boundaries)

# 定义不同的 Neumann 条件
g1 = Expression("-pi*cos(pi*x[0])*sin(pi*x[1])", degree=1)  # 左边界
g2 = Expression("pi*cos(pi*x[0])*sin(pi*x[1])", degree=1)   # 右边界
g3 = Expression("pi*sin(pi*x[0])*cos(pi*x[1])", degree=1)   # 上边界
g4 = Expression("-pi*sin(pi*x[0])*cos(pi*x[1])", degree=1)   # 下边界

# 定义方程右端项 f
alpha = 1.0
# q_ex = Constant(1.0)  # 流体速度
u_ex = Expression("1+sin(pi*x[0])*sin(pi*x[1])", degree=1)
q_ex = Expression("1+0.5*sin(pi*x[0])*sin(pi*x[1])", degree=1)
f = Expression("2*pi*pi*sin(pi*x[0])*sin(pi*x[1])", degree=1) + u_ex * q_ex

# 变分形式
# a = inner(grad(u), grad(v)) * dx + q_ex * u * v * dx
# L = f * v * dx + g1 * v * ds(1) + g2 * v * ds(2) + g3 * v * ds(3) + g4 * v * ds(4)# Neumann 
# L = f * v * dx # Dirichlet 边界条件

F = (inner(grad(u), grad(v)) + q_ex * u * v - f * v) * dx - (g1 * v * ds(1) + g2 * v * ds(2) + g3 * v * ds(3) + g4 * v * ds(4))  # 方程的弱形式
# 计算解
# bc = DirichletBC(V, 1.0, "on_boundary")  # Dirichlet 边界条件
# solve(a == L, u_sol, bc)  # 使用 Dirichlet 边界条件
solve(F == 0, u)  # 使用 Neumann 边界条件



# 可视化
import matplotlib.pyplot as plt
from matplotlib import cm  # 用于自定义颜色映射
print(u.vector())  # 打印解的数值
plot_u = plot(u, cmap=cm.viridis)  # 使用 Viridis 颜色映射
plt.colorbar(plot_u)  # 添加颜色条
plt.title("Solution Heatmap")  # 添加标题
plt.savefig("solution.png")
plt.show()
