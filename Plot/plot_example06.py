import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
import os
import pandas as pd

def tmp(x, y):
    return np.exp(-9.0 * (x - 0.5) ** 2 - 9.0 * (y - 0.5) ** 2)
def q_dagger(x, y): 
    return 1.0 + 0.5 * (x >= y) * tmp(x, y)
def u_dagger(x, y):
    return 1.0 + np.sin(np.pi * x) * np.sin(np.pi * y)

def relative_error(solution, solution_ref):
    err = solution - solution_ref
    relative_err = np.linalg.norm(err) / np.linalg.norm(solution_ref)
    return relative_err

def format_func(value, tick_number):
    return f'{value:.1e}'  # 科学计数法格式

def plot_subplots(values, idxs, title, filename, vmin, vmax):
    num_plots = len(idxs)
    if title in [r'$u^{\dagger}$', r'$q^{\dagger}$']:
        is_first_line = True
        height = 4
    else:
        is_first_line = False
        height = 4 * 0.88 / 0.96
    fig = plt.figure(figsize=(4 * num_plots, height))
    gs = gridspec.GridSpec(1, 6, width_ratios=[0.05, 1, 1, 1, 1, 0.05])  # 最后一列留给 colorbar
    imgs = []
    # idxs[0] = '1'

    ax_title = fig.add_subplot(gs[0])
    ax_title.axis('off')  # 不显示坐标轴
    ax_title.text(1.0, 0.5, title,
                fontsize=24, fontweight='bold', va='center', ha='center', rotation=90)

    for i in range(num_plots):
        ax = fig.add_subplot(gs[i+1])

        img = ax.imshow(values[i], cmap='coolwarm', aspect='auto', vmin=vmin, vmax=vmax)
        ax.invert_xaxis() 
        ax.set_xticks([])  # 去掉X轴刻度
        ax.set_yticks([])  # 去掉Y轴刻度

        _, delta = idxs[i].split('_')
        if delta == '01':
            delta = '1'

        if is_first_line:
            title_fig = '$\delta$=' + delta + '%'
            ax.set_title(title_fig, fontsize=24)  # 设置子图标题
        imgs.append(img)

    cbar_ax = fig.add_subplot(gs[-1])  # 占两行，右侧整列
    cbar = fig.colorbar(imgs[0], cax=cbar_ax)
    if title in [r'$|u_{\text{pinn}}-u^{\dagger}|$', r'$|q_{\text{pinn}}-q^{\dagger}|$',
                  r'$|u_{\text{fem}}-u^{\dagger}|$', r'$|q_{\text{fem}}-q^{\dagger}|$']:
        cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_func))
    cbar.ax.tick_params(labelsize=12)  # 设置颜色条刻度字体大小
    # plt.tight_layout()  # 自动调整子图布局
    top = 0.9 if is_first_line else 0.98
    fig.subplots_adjust(left=0.02, right=0.95, top=top, bottom=0.02, hspace=0.0, wspace=0.1)
    plt.savefig(filename, format='pdf', dpi=1200)
    plt.show()
    plt.close()

#文件名
exp_idx = '01'
idxs_nn = ['09_01', '09_10', '07_20', '09_50']
files = ['udag', 'qdag', 'unn', 'qnn', 'unn_err', 'qnn_err', 'ufem', 'q_fem', 'ufem_err', 'qfem_err']
titles = [r'$u^{\dagger}$', r'$q^{\dagger}$', r'$u_{\text{pinn}}$', r'$q_{\text{pinn}}$',
           r'$|u_{\text{pinn}}-u^{\dagger}|$', r'$|q_{\text{pinn}}-q^{\dagger}|$', r'$u_{\text{fem}}$', 
           r'$q_{\text{fem}}$', r'$|u_{\text{fem}}-u^{\dagger}|$', r'$|q_{\text{fem}}-q^{\dagger}|$']
vmins = [1.0, 0.98, 1.0, 0.98, 0.0, 0.0, 1.0, 0.98, 0.00, 0.0]
vmaxs = [2.0, 1.52, 2.0, 1.52, 0.005, 0.2, 2.0, 1.52, 0.005, 0.2]
lamda = '07'
idxs = ['09_01', '06_10', '05_20', '05_50']

# 设置保存目录路径
directory = './Figures/discontinuous'

# 确保目录存在
if not os.path.exists(directory):
    os.makedirs(directory, exist_ok=True)

# 加载数据
len_ = 501
index = '50000'

q_dags, u_dags, q_nns, u_nns, qnn_errs, unn_errs, q_fems, u_fems, qfem_errs, ufem_errs = [], [], [], [], [], [], [], [], [], []

for i in range(len(idxs_nn)):
    # 使用绝对路径
    file_path = f'./TikPINN_fullobs/output/discontinuous/result_{idxs_nn[i]}/result50000.txt'

    results_nn = pd.read_csv(file_path, header=None, sep=',').values

    x = results_nn[:, 0]
    y = results_nn[:, 1]
    X = x.reshape(len_, len_)
    Y = y.reshape(len_, len_)
    q_dag = q_dagger(X, Y)
    u_dag = u_dagger(X, Y)
    q_nn = results_nn[:, 2].reshape(len_, len_)
    u_nn = results_nn[:, 3].reshape(len_, len_)

    q_dags.append(q_dag)
    q_nns.append(q_nn)
    qnn_errs.append(np.abs(q_nn - q_dag))
    u_dags.append(u_dag)
    u_nns.append(u_nn)
    unn_errs.append(np.abs(u_nn - u_dag))

    file_path = f'./Dolfin-adjoint/discontinuous/result_{idxs[i]}/result.txt'
    results_fem = pd.read_csv(file_path, header=None, sep=',').values
    q_fem = np.flip(results_fem[:, 2].reshape(len_, len_))
    u_fem = np.flip(results_fem[:, 3].reshape(len_, len_))

    q_fems.append(q_fem)
    qfem_errs.append(np.abs(q_fem - q_dag))
    u_fems.append(u_fem)
    ufem_errs.append(np.abs(u_fem - u_dag))

# 使用子图绘制
for k in range(len(titles)):
    values = [u_dags, q_dags, u_nns, q_nns, unn_errs, qnn_errs, u_fems, q_fems, ufem_errs, qfem_errs]
    filename = os.path.join(directory, f'{files[k]}.pdf')
    plot_subplots(values[k], idxs_nn, titles[k], filename, vmins[k], vmaxs[k])

# err1 = relative_error(u_nn, u_dag)
# print(f"u relative error = {err1:.4e}")

# err2 = relative_error(q_nn, q_dag)
# print(f"q relative error = {err2:.4e}")