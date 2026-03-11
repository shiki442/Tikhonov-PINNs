import torch
import torch.nn as nn
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import os

sys.path.append('D:/OneDrive/PC/Code Library/Tikhonov-PINNs')
sys.path.append('C:/Users/78103/OneDrive/PC/Code Library/Tikhonov-PINNs/')
from TikPINN.model.nn import MLP


def q_dagger(x1, x2, x3, x4, x5):
    return 1.0 - (x1 - 0.5) ** 2 - (x2 - 0.5) ** 2 + np.cos(np.pi * (x3 + 1.5)) + np.cos(np.pi * (x4 + 1.5)) + np.cos(np.pi * (x5 + 1.5))


def u_dagger(x1, x2, x3, x4, x5):
    return x1 + x1**3 / 3.0 + x2 + x2**3 / 3.0 + x3 + x3**3 / 3.0 + x4 + x4**3 / 3.0 + x5 + x5**3 / 3.0


if __name__ == '__main__':
    fixed_val = 0.5
    file_path = rf'C:/Users/78103/OneDrive/PC/Code Library/Tikhonov-PINNs/TikPINN/output/fivedim/box8/result_09_00/result50000.txt'

    results_nn = pd.read_csv(file_path, header=None, sep=',').values

    x1 = results_nn[:, 0]
    x2 = results_nn[:, 1]
    x3 = results_nn[:, 2]
    x4 = results_nn[:, 3]
    x5 = results_nn[:, 4]

    tolerance = 1e-6
    mask = (np.abs(x3 - fixed_val) < tolerance) & \
            (np.abs(x4 - fixed_val) < tolerance) & \
            (np.abs(x5 - fixed_val) < tolerance)
    
    # 获取满足条件的数据
    x1_slice = x1[mask]
    x2_slice = x2[mask]
    # 重构网格
    X = x1_slice.reshape(int(np.sqrt(len(x1_slice))), -1)
    Y = x2_slice.reshape(int(np.sqrt(len(x2_slice))), -1)

    q_dag = q_dagger(X, Y, fixed_val, fixed_val, fixed_val)
    u_dag = u_dagger(X, Y, fixed_val, fixed_val, fixed_val)

    q_nn = results_nn[mask, 5].reshape(X.shape)  # 假设第5列是q_nn
    u_nn = results_nn[mask, 6].reshape(X.shape)  # 假设第6列是u_nn
    err_q = np.abs(q_nn - q_dag)
    err_u = np.abs(u_nn - u_dag)

    vmin, vmax = 3.4, 4.02
    fig = plt.figure(figsize=(16, 4))
    gs = gridspec.GridSpec(1, 6, width_ratios=[0.05, 0.00, 1, 1, 1, 0.05], wspace=0.25)

    # ax_title = fig.add_subplot(gs[0])
    # ax_title.axis('off')  # Hide axis
    # ax_title.text(1.0, 0.5, r'$\delta=0$', fontsize=24, fontweight='bold', va='center', ha='center', rotation=90)

    ax_q_dag = fig.add_subplot(gs[2])
    im_main = ax_q_dag.imshow(q_dag, extent=[0, 1, 0, 1], cmap='coolwarm', aspect='auto', vmin=vmin, vmax=vmax)
    # ax_q_dag.set_xlabel(r'$x_1$', fontsize=18)
    # ax_q_dag.set_ylabel(r'$x_2$', fontsize=18)
    ax_q_dag.set_title(r'$q^{\dagger}(x_1, x_2, 0.5, 0.5, 0.5$)', fontsize=18)
    ax_q = fig.add_subplot(gs[3])

    img = ax_q.imshow(q_nn, extent=[0, 1, 0, 1], cmap='coolwarm', aspect='auto', vmin=vmin, vmax=vmax)
    # ax_q.set_xlabel(r'$x_1$', fontsize=18)
    # ax_q.set_ylabel(r'$x_2$', fontsize=18)
    ax_q.set_title(r'$q_{pinn}(x_1, x_2, 0.5, 0.5, 0.5$)', fontsize=18)

    cbar_main = fig.colorbar(
        im_main,
        cax=fig.add_subplot(gs[0]), # 绑定到左侧 colorbar 区域
        orientation='vertical' # 垂直方向（与原代码一致）
    )
    # cbar_main.ax.tick_params(labelsize=12) # 调整 colorbar 刻度字号
    # cbar_main.set_label (r'q ', fontsize=14, labelpad=10)

    ax_err = fig.add_subplot(gs[4])
    im_err = ax_err.imshow(err_q, extent=[0, 1, 0, 1], cmap='coolwarm', aspect='auto', vmin=0.0, vmax=0.5)
    # ax_err.set_xlabel(r'$x_1$', fontsize=18)
    # ax_err.set_ylabel(r'$x_2$', fontsize=18)
    ax_err.set_title(r'$q^{\dagger}-q_{pinn}$', fontsize=18)

    # cbar_ax = fig.add_subplot(gs[-1])  # Last column for colorbar
    cbar_err = fig.colorbar(
        im_err,
        cax=fig.add_subplot(gs[-1]), # 绑定到右侧 colorbar 区域
        orientation='vertical'
    )
    # cbar_err.ax.tick_params(labelsize=12)
    plt.tight_layout()
    plt.show()
