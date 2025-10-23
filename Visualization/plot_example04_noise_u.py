import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
import os
import pandas as pd
from matplotlib.ticker import ScalarFormatter
from scipy.signal import savgol_filter

print(os.getcwd())


def ms(residual):
    return np.mean(np.square(residual))


def mse(pred, target):
    return ms(pred - target)


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


def target_errq(x):
    return 0.012 * np.sqrt(x)


def target_erru(x):
    return 0.0055 * x


def format_func(value, tick_number):
    return f'{value:.1e}'  # Scientific notation format


if __name__ == "__main__":
    len_ = 501
    noise_levels = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

    lambdas = [1.0e-7, 1.0e-8, 1.0e-9, 0.0]
    n_tasks = len(noise_levels)
    err_q = np.zeros((n_tasks, len(lambdas)))
    err_u = np.zeros((n_tasks, len(lambdas)))

    fig, ax = plt.subplots(figsize=(12, 9))

    colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444']
    markers = ['o', 's', '^', 'D']
    line_styles = ['-', '-', '-', '-']

    err_q_list = []
    err_u_list = []
    u_slopses = []
    labels = [r'$\lambda=1.0\times 10^{-7}$', r'$\lambda=1.0\times 10^{-8}$', r'$\lambda=1.0\times 10^{-9}$', r'$\lambda=0.0$']
    for j in range(len(lambdas)):
        lamb = lambdas[j]
        for i in range(len(noise_levels)):
            noise = noise_levels[i]
            task_str = params2str(lamb, noise)
            file_path = f'./TikPINN/output/one_peak_1d_noise/result_{task_str}/result00500.txt'
            results_nn = pd.read_csv(file_path, header=None, sep=',').values
            x = results_nn[:, 0]
            q_dag = q_dagger(x)
            u_dag = u_dagger(x)
            q_nn = results_nn[:, 1]
            u_nn = results_nn[:, 2]
            err_q[i, j] = np.sqrt(mse(q_nn, q_dag) / ms(q_dag))
            err_u[i, j] = np.sqrt(mse(u_nn, u_dag) / ms(u_dag))
        err_q_list.append(err_q[:, j])
        err_u_list.append(err_u[:, j])
        log_noise = np.log10(noise_levels)
        log_err_u = np.log10(err_u[:, j])
        slope_u, _ = np.polyfit(log_noise, log_err_u, 1)
        u_slopses.append(slope_u)
    # smoothed_err_q_list = []
    # for y_data in err_q_list:
    #     # 使用Savitzky-Golay滤波进行光滑化
    #     # window_length为窗口大小（奇数），polyorder为多项式阶数
    #     smoothed = savgol_filter(y_data, window_length=5, polyorder=2)
    #     smoothed_err_q_list.append(smoothed)
    print(u_slopses, np.mean(u_slopses))
    for i, (y_data, color, marker, label) in enumerate(zip(err_u_list, colors, markers, labels)):
        ax.plot(
            noise_levels,
            y_data,
            marker=marker,
            markersize=12,
            markerfacecolor=color,
            markeredgecolor='white',
            markeredgewidth=1.5,
            color=color,
            linewidth=2,
            linestyle=line_styles[i],
            alpha=0.7,
            label=label,
        )
    ax.tick_params(labelsize=15, which='both')
    ax.set_xscale('log')
    ax.set_yscale('log')
    x = np.linspace(0.088, 0.55, len_)
    y = target_erru(x)
    ax.plot(x, y, linestyle='--', color='grey', label=r'$\log\epsilon_u=\log\delta + C$', linewidth=4)

    # log_noise_levels = [np.log(n) for n in noise_levels]
    # log_err_q = np.log(err_q)
    # slope, intercept = np.polyfit(log_noise_levels, log_err_q, 1)
    # trend_line = x ** np.mean(slope) * np.exp(np.mean(intercept))
    # ax.plot(x, trend_line, color='#6366f1', label=r'$\epsilon=C\delta^{0.757}$', linewidth=3.5, linestyle='--', alpha=0.9)

    ax.grid(True, which="both", linestyle='--', alpha=0.3, color='gray')
    ax.legend(loc='upper left', fontsize=20, frameon=True, facecolor='white', edgecolor='#dddddd', framealpha=0.9, borderaxespad=1)
    ax.set_xlabel(r'Noise level $\delta$ (log scale)', fontsize=27)
    ax.set_ylabel(r'Relative $L_2$ error $\epsilon_u$ (log scale)', fontsize=27)
    ax.set_title(r'Effect of Noise Level $\delta$ on Prediction Error $\epsilon_u$', fontsize=27)
    # # 修改坐标轴刻度格式设置部分
    ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False, useMathText=False))

    # # 对于次要刻度，也可以添加类似设置
    ax.xaxis.set_minor_formatter(ScalarFormatter(useOffset=False, useMathText=False))

    plt.tight_layout()
    plt.show()
