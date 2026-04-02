import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
import os

print(os.getcwd())

def q_dagger(x, y):
    # Non-smooth radial function
    # RegionExpression: piecewise function based on radial distance from (0.4, 0.4)
    r = np.sqrt((x - 0.4)**2 + (y - 0.4)**2)
    r_inner = 0.15
    r_outer = 0.2

    q = np.ones_like(r) * 1.0  # outer region: 1.0
    mask_inner = r <= r_inner
    mask_middle = (r > r_inner) & (r < r_outer)

    q[mask_inner] = 5.0
    q[mask_middle] = 5.0 + (1.0 - 5.0) * (r[mask_middle] - r_inner) / (r_outer - r_inner)

    return q

def u_dagger(x, y):
    # u_exact = 4*x*(1-x)*y*(1-y)
    return 4 * x * (1 - x) * y * (1 - y)


if __name__ == "__main__":
    len_ = 501

    # Lambda = 1e-9 (09), noise levels: 1%, 10%
    str_params = ['09_01', '09_10']
    q_nns, u_nns = [], []
    titles_q = [r'$q^{\dagger}$ (True)', r'$\delta = 1$%', r'$\delta = 10$%']
    titles_u = [r'$u^{\dagger}$ (True)', r'$\delta = 1$%', r'$\delta = 10$%']

    directory = './Figures/non_smooth_radial'
    if not os.path.exists(directory):
        os.makedirs(directory)

    for i in range(len(str_params)):
        file_path = f'./output/non_smooth_radial/result_{str_params[i]}/result.txt'

        results_nn = pd.read_csv(file_path, header=None, sep=',').values

        x = results_nn[:, 0]
        y = results_nn[:, 1]
        # Data is column-major (x fixed, y varies), so reshape then transpose
        X = x.reshape(len_, len_).T
        Y = y.reshape(len_, len_).T
        q_dag = q_dagger(X, Y)
        u_dag = u_dagger(X, Y)
        q_nn = results_nn[:, 2].reshape(len_, len_).T
        u_nn = results_nn[:, 3].reshape(len_, len_).T

        q_nns.append(q_nn)
        u_nns.append(u_nn)

    q_nns.append(q_dag)
    u_nns.append(u_dag)

    # 3 plots: 1 reference + 2 results
    num_cols = 3  # True, 1% noise, 10% noise
    fig = plt.figure(figsize=(4 * num_cols, 4 * 2))
    gs = gridspec.GridSpec(2, num_cols + 1, width_ratios=[1, 1, 1, 0.05])
    imgs = []

    # For q: range is 1.0 to 5.0
    vmin_q, vmax_q = 1.0, 5.0
    extent = [0, 1, 0, 1]  # [xmin, xmax, ymin, ymax]
    ax = fig.add_subplot(gs[0, 0])
    img = ax.imshow(q_nns[-1], cmap='coolwarm', aspect='auto', vmin=vmin_q, vmax=vmax_q, origin='lower', extent=extent)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(titles_q[0], fontsize=18, pad=10)

    for i in range(2):
        ax = fig.add_subplot(gs[0, i + 1])
        img = ax.imshow(q_nns[i], cmap='coolwarm', aspect='auto', vmin=vmin_q, vmax=vmax_q, origin='lower', extent=extent)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(titles_q[i + 1], fontsize=18, pad=10)
        imgs.append(img)

    cbar_ax = fig.add_subplot(gs[0, -1])
    cbar = fig.colorbar(imgs[0], cax=cbar_ax)
    cbar.ax.tick_params(labelsize=12)

    imgs = []
    # For u: range is 0.0 to 0.25
    vmin_u, vmax_u = 0.0, 0.25
    ax = fig.add_subplot(gs[1, 0])
    img = ax.imshow(u_nns[-1], cmap='coolwarm', aspect='auto', vmin=vmin_u, vmax=vmax_u, origin='lower', extent=extent)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(titles_u[0], fontsize=18, pad=10)

    for i in range(2):
        ax = fig.add_subplot(gs[1, i + 1])
        img = ax.imshow(u_nns[i], cmap='coolwarm', aspect='auto', vmin=vmin_u, vmax=vmax_u, origin='lower', extent=extent)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(titles_u[i + 1], fontsize=18, pad=10)
        imgs.append(img)

    cbar_ax = fig.add_subplot(gs[1, -1])
    cbar = fig.colorbar(imgs[0], cax=cbar_ax)
    cbar.ax.tick_params(labelsize=12)

    fig.subplots_adjust(left=0.02, right=0.95, top=0.9, bottom=0.02, hspace=0.15, wspace=0.1)

    filename_q = os.path.join(directory, 'ex05_heatmap_qfem.png')
    filename_u = os.path.join(directory, 'ex05_heatmap_ufem.png')
    plt.savefig(os.path.join(directory, 'ex05_heatmap_fem.png'), format='png', dpi=150, bbox_inches='tight')
    plt.show()
