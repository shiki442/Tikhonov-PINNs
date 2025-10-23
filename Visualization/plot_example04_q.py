import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import numpy as np
import pandas as pd
import seaborn as sns


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


def q_dagger(x):
    return np.exp(x)


def u_dagger(x):
    return 1.0 + np.sin(np.pi * x)


def smooth_data(y, window_size=5):
    window = np.ones(window_size) / window_size
    return np.convolve(y, window, mode='same')


lamb = '07'
deltas = ['0', '20', '50']
fig = plt.figure(figsize=(19, 5))
gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])
ax0 = fig.add_subplot(gs[0])
ax1 = fig.add_subplot(gs[1])
ax2 = fig.add_subplot(gs[2])
axes = [ax0, ax1, ax2]
noise_levels = ['00', '20', '50']
for i in range(noise_levels.__len__()):
    file_path = f'./TikPINN/output/one_peak_1d_noise/result_{lamb}_{noise_levels[i]}/result00500.txt'
    results_nn = pd.read_csv(file_path, header=None, sep=',').values
    x = results_nn[:, 0]
    y = results_nn[:, 1]

    q_dag = q_dagger(x)
    u_dag = u_dagger(x)
    err = np.abs(q_dag - y)
    ax = axes[i]
    ax.plot(x, q_dag, label=r'$q^\dagger$', color='#ef4444', linewidth=2)
    ax.plot(x, y, label=r'$q_{pinn}$', color='#000000', linewidth=2, linestyle=':')
    ax.set_title(r'noise level: $\delta=$' + f'{deltas[i]}%', fontsize=20)
    ax.set_xlabel('x', fontsize=20)
    ax.set_ylabel('q(x)', fontsize=20)
    ax.legend(fontsize=15)
    ax.grid(True, which="both", linestyle='--', alpha=0.3, color='gray')

    ax2 = ax.twinx()
    ax2.plot(x, smooth_data(err, window_size=15), label=r'$|q^\dagger - q_{pinn}|$', color='#7f8c8d', linewidth=2, linestyle='--')
    ax2.set_ylabel(r'$|q^\dagger - q_{pinn}|$', fontsize=20)
    ax2.set_ylim(0, 0.1)
    ax2.legend(fontsize=15)
plt.tight_layout()
# plt.savefig(f'./Figures/compare_reg_q.eps', format='eps', dpi=1200)
plt.show()


# ax0.legend()
# ax1.legend()
# plt.savefig(f'./Figures/err_q.eps', format='eps', dpi=1200)
# plt.show()
