import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from sklearn.metrics import mean_squared_error


# 假设这些函数在原始代码中已定义
def params2str(seed, n):
    # Convert parameters to string
    str = f"{n}_{seed}"
    return str


def u_dagger(x):
    return 1.0 + np.sin(np.pi * x)


def mse(pred, true):
    return mean_squared_error(true, pred)


def ms(data):
    return np.mean(data**2)


n_list = [10, 100, 500, 1000, 5000, 10000, 50000]  # Noise levels in percentage

seeds = [41, 42, 43, 58, 47]
# 参数设置
n_tasks = len(n_list)
n_lambdas = len(seeds)

# 初始化误差存储数组
err_u = np.zeros((n_tasks, n_lambdas))

# 设置绘图样式
fig, ax = plt.subplots(figsize=(12, 7))
# 原始线的样式（较淡）
original_colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6']  # 新增紫色
original_markers = ['o', 's', '^', 'D', 'v']  # 新增倒三角形标记
original_line_styles = ['-', '--', '-.', ':', '-.']  # 新增点划线型
original_alpha = 0.3  # 保持淡色透明度，不干扰主线

# 主线样式（不变，突出显示）
main_color = '#6366f1'  # 靛蓝色
main_marker = 'o'
main_alpha = 0.9

# 读取数据并计算误差
for i in range(n_tasks):
    n = n_list[i]
    for j in range(n_lambdas):
        seed = seeds[j]
        # 模拟数据（实际使用时替换为文件读取）
        task_str = params2str(seed, n)
        file_path = f'./TikPINN/output/one_peak_1d_nsample/result_{task_str}/result00500.txt'
        results_nn = pd.read_csv(file_path, header=None, sep=',').values
        x = results_nn[:, 0]
        u_dag = u_dagger(x)
        u_nn = results_nn[:, 2]
        err_u[i, j] = np.sqrt(mse(u_nn, u_dag) / ms(u_dag))


# 绘制淡色的原始线
for j in range(n_lambdas):
    ax.plot(
        n_list,
        err_u[:, j],
        marker=original_markers[j],
        markersize=8,
        markerfacecolor=original_colors[j],
        markeredgecolor='white',
        markeredgewidth=1,
        color=original_colors[j],
        linewidth=1.5,
        linestyle=original_line_styles[j],
        alpha=original_alpha,
    )

# 合并所有数据计算总体统计量
err_q_combined = err_u.reshape(n_tasks, -1)
err_mean = np.mean(err_q_combined, axis=1)
err_std = np.std(err_q_combined, axis=1)

# 绘制合并后的主线
(main_line,) = ax.plot(
    n_list,
    err_mean,
    marker=main_marker,
    markersize=12,
    markerfacecolor=main_color,
    markeredgecolor='white',
    markeredgewidth=1.5,
    color=main_color,
    linewidth=3,
    alpha=main_alpha,
)

# 绘制置信区间
ci_fill = ax.fill_between(
    n_list,
    err_mean - 2 * err_std,
    err_mean + 2 * err_std,
    color=main_color,
    alpha=0.2,
    linewidth=0,
    label=r'Confidence Interval (±2$\sigma$)',
)

# 设置坐标轴为对数刻度
ax.set_xscale('log')
ax.set_yscale('log')

# 添加网格线
ax.grid(True, which="both", linestyle='--', alpha=0.3, color='gray')

# 设置图例 - 组合原始线标签和主线标签
handles, labels = ax.get_legend_handles_labels()
# 创建一个"原始数据"的综合标签
original_handle = plt.Line2D([], [], color='gray', alpha=original_alpha, linewidth=1.5, label='Individual λ Curves')
# 重新组织图例
handles = [original_handle] + handles[-2:]  # 原始数据综合标签 + 主线 + 置信区间
labels = ['Combined Mean', r'Confidence Interval (±2$\sigma$)']

ax.legend(
    handles, labels, loc='upper right', fontsize=14, frameon=True, facecolor='white', edgecolor='#dddddd', framealpha=0.9, borderaxespad=1
)

# 设置坐标轴标签和标题
ax.set_xlabel(r'num of samples $n_{\text{samples}}$ (log scale)', fontsize=16)
ax.set_ylabel(r'Relative $L_2$ error $\epsilon_u$ (log scale)', fontsize=16)
ax.set_title(r'Effect of $n_{\text{samples}}$ on Prediction Error $\epsilon_u$', fontsize=18)

plt.tight_layout()
plt.show()
