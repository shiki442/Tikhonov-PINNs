import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

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

def smooth_data(y, window_size=5):
    window = np.ones(window_size) / window_size
    return np.convolve(y, window, mode='same')

idxs = ['07_01', '07_10', '07_20']
fig = plt.figure(figsize=(14, 6))
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
ax0 = fig.add_subplot(gs[0])
ax1 = fig.add_subplot(gs[1])

for i in range(len(idxs)):
    _, delta_str = idxs[i].split('_')
    if delta_str == '01':
        delta_str = '1'
    # Load data
    err_q = np.loadtxt(f"./Dolfin-adjoint/non_smooth/result_{idxs[i]}/err_with_iter.txt")
    # err_q = data['err_q']
    x = np.arange(1, len(err_q) + 1)
    err_q_smoothed = smooth_data(err_q, window_size=1)
    n = len(x) - 10
    ax0.plot(x[:130], err_q_smoothed[:130], label=r'noise level: ' + delta_str + '%', linewidth=2)
    ax0.set_yscale('log')
    ax0.set_ylim(1e-2, 1e1)
    ax0.set_title(r"err($q_{fem}$)", fontsize=16)
    ax0.set_xlabel("Iteration")

    loss_nn = pd.read_csv(f'./TikPINN_fullobs/output/non_smooth/result_{idxs[i]}/loss.txt', header=None, sep=',').values
    err_qnn = loss_nn[:, 1]
    err_qnn_smoothed = smooth_data(err_qnn, window_size=1)
    n = len(err_qnn_smoothed)
    x = np.arange(1, n+1)
    ax1.plot(x[:30000], err_qnn_smoothed[:30000], label=r'noise level: ' + delta_str + '%', linewidth=2)
    ax1.set_yscale('log')
    ax1.set_ylim(1e-4, 1e0)
    ax1.set_title(r"err($q_{pinn}$)", fontsize=16)
    ax1.set_xlabel("Iteration")

    fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, hspace=0.0, wspace=0.1)

ax0.legend()
ax1.legend() 
plt.savefig(f'./Figures/err_q.pdf', format='pdf', dpi=1200)
plt.show()
