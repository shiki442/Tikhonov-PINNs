import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
import os
import pandas as pd

print(os.getcwd())

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

def q_dagger(x, y):
    return 1.0 + 0.5 * np.sin(np.pi * x) * np.sin(np.pi * y)

def u_dagger(x, y):
    return 1.0 + np.sin(np.pi * x) * np.sin(np.pi * y)

def format_func(value, tick_number):
    return f'{value:.1e}'  # Scientific notation format

def plot_subplots(values, str_params, title=None, filename=None, vmin=None, vmax=None):
    num_plots = 5
    height = 3.7
    fig = plt.figure(figsize=(4 * num_plots, height))
    gs = gridspec.GridSpec(1, 7, width_ratios=[1, 0.3, 1, 1, 1, 1, 0.05])  # The last column is reserved for the colorbar
    imgs = []

    ax = fig.add_subplot(gs[0])
    img = ax.imshow(values[-1], cmap='coolwarm', aspect='auto', vmin=vmin, vmax=vmax)
    ax.invert_xaxis() 
    ax.set_xticks([])  # Remove x-axis ticks
    ax.set_yticks([])  # Remove y-axis ticks
    ax.set_title(title, fontsize=24)

    for i in range(4):
        ax = fig.add_subplot(gs[i+2])

        img = ax.imshow(values[i], cmap='coolwarm', aspect='auto', vmin=vmin, vmax=vmax)
        ax.invert_xaxis() 
        ax.set_xticks([])  # Remove x-axis ticks
        ax.set_yticks([])  # Remove y-axis ticks

        _, delta = str_params[i].split('_')
        if delta == '01':
            delta = '1'
        title_fig = '$\delta$=' + delta + '%'
        ax.set_title(title_fig, fontsize=24)
        imgs.append(img)

    cbar_ax = fig.add_subplot(gs[-1])  # Use the last column for the colorbar
    cbar = fig.colorbar(imgs[0], cax=cbar_ax)
    cbar.ax.tick_params(labelsize=12)  # Set the font size for the colorbar ticks

    fig.subplots_adjust(left=0.02, right=0.95, top=0.85, bottom=0.02, hspace=0.0, wspace=0.1)
    plt.savefig(filename, format='pdf', dpi=1200)
    plt.show()


if __name__ == "__main__":
    len_ = 501

    str_params = ['09_01', '08_10', '09_20', '07_50']
    q_dags, u_dags, q_nns, u_nns = [], [], [], []
    files = ['udag', 'qdag', 'unn', 'qnn']
    titles = [r'$u^{\dagger}$', r'$q^{\dagger}$', r'$u_{\text{pinn}}$', r'$q_{\text{pinn}}$']

    directory = './Figures/one_peak'
    if not os.path.exists(directory):
        os.makedirs(directory)

    for i in range(len(str_params)):
        file_path = f'./TikPINN/output/one_peak/result_{str_params[i]}/result50000.txt'

        results_nn = pd.read_csv(file_path, header=None, sep=',').values

        x = results_nn[:, 0]
        y = results_nn[:, 1]
        X = x.reshape(len_, len_)
        Y = y.reshape(len_, len_)
        q_dag = q_dagger(X, Y)
        u_dag = u_dagger(X, Y)
        q_nn = results_nn[:, 2].reshape(len_, len_)
        u_nn = results_nn[:, 3].reshape(len_, len_)

        q_nns.append(q_nn)
        u_nns.append(u_nn)

    q_nns.append(q_dag)
    u_nns.append(u_dag)

    filename = os.path.join(directory, f'qnn.pdf')
    plot_subplots(q_nns, str_params, r'$q^{\dagger}$', filename, 1.0, 1.5)
    filename = os.path.join(directory, f'unn.pdf')
    plot_subplots(u_nns, str_params, r'$u^{\dagger}$', filename, 1.0, 2.0)