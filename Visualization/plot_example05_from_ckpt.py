import torch
import torch.nn as nn
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import os

# 网络所在的路径
sys.path.append('C:/Users/78103/OneDrive/PC/Code Library/Tikhonov-PINNs/')
from TikPINN.model.nn import MLP


def q_dagger(x1, x2, x3, x4, x5):
    return 1.0 - (x1 - 0.5) ** 2 - (x2 - 0.5) ** 2 + np.cos(np.pi * (x3 + 1.5)) + np.cos(np.pi * (x4 + 1.5)) + np.cos(np.pi * (x5 + 1.5))


def u_dagger(x1, x2, x3, x4, x5):
    return x1 + x1**3 / 3.0 + x2 + x2**3 / 3.0 + x3 + x3**3 / 3.0 + x4 + x4**3 / 3.0 + x5 + x5**3 / 3.0


if __name__ == '__main__':
    # load model的部分自行替换一下
    model = torch.load('./TikPINN/output/fivedim/box8/result_09_10/checkpoint_epoch50000.pth')
    q = MLP(5, 1, [26, 26, 26, 26], [0.48, 8], nn.Tanh())
    q.load_state_dict(model['q_net_state_dict'])
    u = MLP(5, 1, [26, 26, 26, 26], [0.48, 8], nn.Tanh())
    u.load_state_dict(model['u_net_state_dict'])
    q.eval()
    u.eval()

    fixed = [0.5, 0.5, 0.5]
    x1 = np.linspace(0, 1, 100)
    x2 = np.linspace(0, 1, 100)
    X1, X2 = np.meshgrid(x1, x2)
    points = np.stack([X1, X2, np.full_like(X1, fixed[0]), np.full_like(X1, fixed[1]), np.full_like(X1, fixed[2])], axis=-1)
    points_flat = points.reshape(-1, 5)

    points_tensor = torch.tensor(points_flat, dtype=torch.float32)
    q_dag = q_dagger(points_flat[:, 0], points_flat[:, 1], points_flat[:, 2], points_flat[:, 3], points_flat[:, 4]).reshape(X1.shape)
    u_dag = u_dagger(points_flat[:, 0], points_flat[:, 1], points_flat[:, 2], points_flat[:, 3], points_flat[:, 4]).reshape(X1.shape)
    with torch.no_grad():
        q_nn = q(points_tensor).cpu().numpy().reshape(X1.shape)
        u_nn = u(points_tensor).cpu().numpy().reshape(X1.shape)
    err_q = np.abs(q_nn - q_dag)
    err_u = np.abs(u_nn - u_dag)

    vmin, vmax = 3.5, 4.0
    fig = plt.figure(figsize=(16, 4))
    gs = gridspec.GridSpec(1, 7, width_ratios=[0.05, 0.05, 0.00, 1, 1, 1, 0.05], wspace=0.25)

    ax_title = fig.add_subplot(gs[0])
    ax_title.axis('off')  # Hide axis
    ax_title.text(1.0, 0.5, r'$\delta=10\%$', fontsize=24, fontweight='bold', va='center', ha='center', rotation=90)

    ax_q_dag = fig.add_subplot(gs[3])
    im_main = ax_q_dag.imshow(q_dag, extent=[0, 1, 0, 1], cmap='coolwarm', aspect='auto', vmin=vmin, vmax=vmax)
    # ax_q_dag.set_xlabel(r'$x_1$', fontsize=18)
    # ax_q_dag.set_ylabel(r'$x_2$', fontsize=18)
    ax_q_dag.set_title(r'$q^{\dagger}(x_1, x_2, 0.5, 0.5, 0.5$)', fontsize=19)
    ax_q = fig.add_subplot(gs[4])

    img = ax_q.imshow(q_nn, extent=[0, 1, 0, 1], cmap='coolwarm', aspect='auto', vmin=vmin, vmax=vmax)
    # ax_q.set_xlabel(r'$x_1$', fontsize=18)
    # ax_q.set_ylabel(r'$x_2$', fontsize=18)
    ax_q.set_title(r'$q_{pinn}(x_1, x_2, 0.5, 0.5, 0.5$)', fontsize=19)

    cbar_main = fig.colorbar(
        im_main,
        cax=fig.add_subplot(gs[1]), # 绑定到左侧 colorbar 区域
        orientation='vertical' # 垂直方向（与原代码一致）
    )
    # cbar_main.ax.tick_params(labelsize=12) # 调整 colorbar 刻度字号
    # cbar_main.set_label (r'q ', fontsize=14, labelpad=10)

    ax_err = fig.add_subplot(gs[5])
    im_err = ax_err.imshow(err_q, extent=[0, 1, 0, 1], cmap='coolwarm', aspect='auto', vmin=0.0, vmax=0.3)
    # ax_err.set_xlabel(r'$x_1$', fontsize=18)
    # ax_err.set_ylabel(r'$x_2$', fontsize=18)
    ax_err.set_title(r'|$q^{\dagger}-q_{pinn}$|', fontsize=19)

    # cbar_ax = fig.add_subplot(gs[-1])  # Last column for colorbar
    cbar_err = fig.colorbar(
        im_err,
        cax=fig.add_subplot(gs[-1]), # 绑定到右侧 colorbar 区域
        orientation='vertical'
    )
    # cbar_err.ax.tick_params(labelsize=12)
    plt.tight_layout()
    plt.show()
