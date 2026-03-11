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


def q_dagger(x):
    return np.exp(x)


def mse(pred, true):
    return mean_squared_error(true, pred)


def ms(data):
    return np.mean(data**2)


net_widths = [32, 40, 48, 64]  # Noise levels in percentage

deltas = [0.1, 0.2, 0.4, 0.5]
# 参数设置
n_tasks = len(net_widths)
n_lambdas = len(deltas)

# 初始化误差存储数组
err_q = np.zeros((n_tasks, n_lambdas))

# 设置绘图样式
fig, ax = plt.subplots(figsize=(12, 9))
# 原始线的样式（较淡）
original_colors = ["#abc9fa", "#a9d1c4", "#ffd690", "#ff9191", "#c2aef0"]  # 新增紫色
original_markers = ['o', 's', '^', 'D', 'v']  # 新增倒三角形标记
original_line_styles = ['-', '--', '-.', ':', '-.']  # 新增点划线型
original_alpha = 1.0  # 保持淡色透明度，不干扰主线

# 主线样式（不变，突出显示）
main_color = '#6366f1' # 靛蓝色
main_marker = 'o'
main_alpha = 1.0

# 读取数据并计算误差
for i in range(n_tasks):
    width = net_widths[i]
    for j in range(n_lambdas):
        delta = deltas[j]
        # 模拟数据（实际使用时替换为文件读取）
        file_path = f'./TikPINN/output/one_peak_1d_w/result_{int(100*delta)}_{width}/result10000.txt'
        results_nn = pd.read_csv(file_path, header=None, sep=',').values
        x = results_nn[:, 0]
        q_dag = q_dagger(x)
        q_nn = results_nn[:, 1]
        err_q[i, j] = np.sqrt(mse(q_nn, q_dag) / ms(q_dag))


# 绘制淡色的原始线
for j in range(n_lambdas):
    ax.plot(
        net_widths,
        err_q[:, j],
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
    ax.tick_params(labelsize=16)
# 合并所有数据计算总体统计量
err_q_combined = err_q.reshape(n_tasks, -1)
err_mean = np.mean(err_q_combined, axis=1)
err_std = np.std(err_q_combined, axis=1)

# 绘制合并后的主线
(main_line,) = ax.plot(
    net_widths,
    err_mean,
    marker=main_marker,
    markersize=12,
    markerfacecolor=main_color,
    markeredgecolor='white',
    markeredgewidth=1.5,
    color=main_color,
    linewidth=3.5,
    alpha=main_alpha,
)

# 绘制置信区间
ci_fill = ax.fill_between(
    net_widths,
    err_mean - 2 * err_std,
    err_mean + 2 * err_std,
    color="#d9d9ff",
    alpha=1.0,
    linewidth=0,
    label=r'Confidence Interval (±2$\sigma$)',
)


# 添加网格线
ax.grid(True, which="both", linestyle='--', alpha=1.0, color='gray')

# 设置图例 - 组合原始线标签和主线标签
handles, labels = ax.get_legend_handles_labels()
# 创建一个"原始数据"的综合标签
original_handle = plt.Line2D([], [], color='gray', alpha=original_alpha, linewidth=1.5, label='Individual λ Curves')
# 重新组织图例
handles = [original_handle] + handles[-2:]  # 原始数据综合标签 + 主线 + 置信区间
labels = ['Combined Mean', r'Confidence Interval (±2$\sigma$)']

ax.legend(
    handles, labels, loc='upper right', fontsize=20, frameon=True, facecolor='white', edgecolor='#dddddd', framealpha=0.9, borderaxespad=1
)

# 设置坐标轴标签和标题
ax.set_xlabel(r'net width', fontsize=27)
ax.set_ylabel(r'Relative $L_2$ error $\epsilon_q$', fontsize=27)
ax.set_title(r'Effect of net width on Prediction Error $\epsilon_q$', fontsize=27)

plt.tight_layout()
plt.show()
