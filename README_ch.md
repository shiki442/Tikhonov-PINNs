# Tikhonov-PINNs

**基于 Tikhonov 正则化物理信息神经网络的势函数识别**

本代码库提供了使用 Tikhonov 正则化物理信息神经网络 (PINNs) 求解反问题的 PyTorch 实现。该方法能够从噪声测量数据中识别椭圆型偏微分方程中的未知势函数系数。

论文链接：[Potential Identification via Tikhonov-PINNs](https://iopscience.iop.org/article/10.1088/1361-6420/ae199a)

## 概述

代码库实现了双网络架构：
- **q_net**: 估计未知的势函数系数 $q(x)$
- **u_net**: 近似 PDE 的解 $u(x)$

该方法求解如下形式的反问题：

$$-\Delta u + q u = f \quad \text{in } \Omega$$

带有 Neumann 边界条件：

$$\frac{\partial u}{\partial n} = g \quad \text{on } \partial\Omega$$

## 仓库结构

```
Tikhonov-PINNs/
├── TikPINN/               # 主 PINN 求解器包
│   ├── main.py            # 训练入口程序
│   ├── tune_hyperparams.py # Optuna 超参数优化
│   ├── submit_jobs.py     # SLURM 批量作业提交
│   ├── run.slurm          # SLURM 提交脚本模板
│   ├── GenerateData/      # 合成数据生成
│   │   ├── generate_data*.py   # 不同示例的数据生成器
│   │   └── problems/      # 问题定义 (Example01-06)
│   ├── model/             # 核心模型组件
│   │   ├── nn.py          # 带残差连接的 MLP 网络
│   │   ├── loss.py        # TikPINN 损失 (测量 + PDE + 正则化)
│   │   ├── data.py        # 数据集和分布式采样器
│   │   ├── optim.py       # Adam/LBFGS 优化器及预热调度器
│   │   ├── train.py       # 三阶段训练流程
│   │   ├── utils.py       # 自动微分工具、范数计算、检查点
│   │   └── problem.py     # PDE 算子和基础问题类
│   ├── config/            # YAML 配置文件
│   └── requirements.txt   # Python 依赖
│
├── TikFEM/                # 基于 FEniCS 的有限元求解器
│   ├── solvepde.py        # 前向 PDE 求解器
│   ├── gaussian_peak.py   # 高斯峰测试问题
│   ├── non_smooth.py      # 非光滑系数情况
│   └── utils.py           # 网格生成和工具函数
│
├── Visualization/         # 结果可视化脚本
│   ├── plot_*.py          # 各种绘图脚本
│
└── CLAUDE.md              # 项目文档
```

## 安装

### TikPINN (基于 PyTorch)

```bash
cd TikPINN
pip install -r requirements.txt
```

**依赖要求：**
- Python >= 3.8
- PyTorch >= 2.0.0
- CUDA 兼容的 GPU (训练推荐使用)

### TikFEM (基于 FEniCS)

```bash
# 安装 FEniCS (平台特定)
# 参见 https://fenicsproject.org/download/

cd TikFEM
pip install -r requirements.txt  # 如果有
```

## 快速开始

### 1. 生成训练数据

```bash
cd TikPINN/GenerateData

# 示例 03: 标准测试问题
python generate_data_example03.py

# 示例 05: 高斯峰
python generate_data_example05.py

# 所有可用示例：
# - example01.py: 2D exp(x) 势函数
# - example02.py: 2D 标准测试问题
# - example03.py: 标准问题 (1D/2D)
# - example04.py: 宽度/噪声敏感性研究
# - example05.py: 高斯峰
# - example06.py: 1D (1+x)*exp(x) 势函数
```

### 2. 配置训练

编辑 `TikPINN/config/params.yml`：

```yaml
task:
  idx: "05"           # 问题 ID
  noise_str: "01"     # 噪声水平 (01=1%, 10=10%)
  dim: 2              # 问题维度

dataloader_params:
  batch_size: 2500
  n_samples: 50000

q_net_params:
  width: 128
  depth: 4
  activation: "tanh"
  box: [1, 5]         # q 的输出范围

u_net_params:
  width: 64
  depth: 5
  activation: "swish"
  box: null

loss_params:
  alpha: 1.0          # 测量损失权重
  lamb: 1.0e-7        # Tikhonov 正则化权重
  regularization: "H2"  # 选项：H2, L2, 0

optim_params_adam:
  q_lr: 1.0e-4
  u_lr: 1.0e-4

train_params:
  pretrain_epochs_u: 100
  num_epochs: [5000, 0]  # [Adam 轮数，LBFGS 轮数]
```

### 3. 运行训练

**单 GPU/CPU:**
```bash
cd TikPINN
python main.py --config_path config/params.yml
```

**多 GPU (DDP):**
```bash
# 自动使用所有可用 GPU
python main.py --config_path config/params.yml
```

**从检查点恢复:**
```bash
# 将 checkpoint.pt 放入 results 目录或在配置中设置 checkpoint_path
python main.py --config_path config/params.yml
```

### 4. 超参数调优

```bash
cd TikPINN
python tune_hyperparams.py --n_trials 50 --config_path config/params.yml
```

### 5. SLURM 集群提交

```bash
# 为超参数扫描生成批量配置
python submit_jobs.py --mode noise_sweep
python submit_jobs.py --mode h2_sweep

# 提交作业
sbatch run.slurm
```

### 6. 可视化

```bash
# 查看训练曲线
tensorboard --logdir TikPINN/logs/

# 绘制结果
cd Visualization
python plot_ex03_fem.py
python plot_heatmap_ex03.py
python plot_training_curves_ex05.py
```

## 方法详解

### 损失函数

总损失由三个部分组成：

$$\mathcal{L}_{total} = \mathcal{L}_{PINNs} + \alpha \mathcal{L}_{measurement} + \lambda \mathcal{L}_{regularization}$$

- **$\mathcal{L}_{PINNs}$**: PDE 残差 $(-\Delta u + qu - f)^2$ + Neumann 边界条件误差
- **$\mathcal{L}_{measurement}$**: 预测值与噪声观测值的均方误差
- **$\mathcal{L}_{regularization}$**: $q$ 的 H2 或 L2 范数 (Tikhonov 正则化)

### 训练流程

训练分为三个阶段：

1. **预训练** (100-1000 轮): 仅使用测量损失训练 u_net
2. **联合训练** (1000-5000 轮): Adam 优化器 + 预热 + 余弦退火
3. **精细调优** (可选，0-100 轮): LBFGS 优化器进行最终微调

### 网络架构

两个网络均使用 MLP，包含：
- 残差连接 (Block 结构)
- 可配置的深度和宽度
- 可选的 sigmoid 投影输出范围限制
- Xavier 初始化

## 数据格式

生成的数据保存为 `.pt` 文件：

```python
{
    'int_points': (n_int, dim),      # 内部点坐标
    'bdy_points': (n_bdy, dim),      # 边界点坐标
    'normal_vec': (n_bdy, dim),      # 外法向量
    'm_int': (n_int, 1),             # 内部噪声测量值
    'm_bdy': (n_bdy, 1),             # 边界噪声测量值
    'f_val': (n_int, 1),             # 源项值
    'g_val': (n_bdy, 1),             # 边界通量值
    'u_dagger': (n_int, 1),          # 精确解
    'q_dagger': (n_int, 1),          # 精确参数
}
```

## 问题索引参考表

| idx | 维度 | 描述 |
|-----|------|------|
| 01  | 2D  | exp(x) 势函数 |
| 02  | 2D  | 标准测试问题 |
| 03  | 1D/2D | 示例问题 |
| 04  | 1D/2D | 宽度/噪声敏感性 |
| 05  | 2D  | 高斯峰 |
| 06  | 1D  | (1+x)*exp(x) 势函数 |

## TensorBoard 日志

训练指标会记录到 TensorBoard：

- **损失曲线**: total, measurement, PINNs, regularization
- **误差曲线**: q_relative_error, u_relative_error
- **热图**: 预测值 vs 精确值 (可配置间隔)

```yaml
train_params:
  heatmap_every_n_epochs: 1000  # 每 N 轮记录热图
```

## 检查点系统

检查点会自动保存：

```yaml
checkpoint_params:
  save_top_k: 3           # 保留最好的 K 个模型
  save_last: true         # 保存最终检查点
  every_n_epochs: 500     # 保存频率
```

将 `checkpoint.pt` 放入 results 目录即可恢复训练。

## API 参考

### 核心组件

```python
from model.nn import get_network
from model.loss import TikPINNLoss
from model.optim import get_optimizer, get_scheduler
from model.train import train
from model.data import get_dataloader
```

### 损失分量

```python
loss = TikPINNLoss(alpha=1.0, lamb=1e-7, regularization='H2')

# 分别获取所有分量
components = loss.get_loss_components(q_net, u_net, samples)
# 返回：{'total', 'measurement', 'pinns', 'regularization'}
```

### 误差度量

```python
from model.loss import relative_error_q, relative_error_u

q_rel_err = relative_error_q(q_net, samples)
u_rel_err = relative_error_u(u_net, samples)
```

## 引用

如果您在研究中使用了本代码，请引用：

```bibtex
@article{TikhonovPINNs2025,
  title={Potential Identification via Tikhonov-PINNs},
  author={},
  journal={Inverse Problems},
  year={2025},
  publisher={IOP Publishing},
  doi={10.1088/1361-6420/ae199a}
}
```

## 联系方式

如有问题或疑问，请在仓库中提交 issue。
