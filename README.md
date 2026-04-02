# Tikhonov-PINNs

**Potential Identification via Tikhonov-Regularized Physics-Informed Neural Networks**

This repository provides a PyTorch implementation for solving inverse problems using Physics-Informed Neural Networks (PINNs) with Tikhonov regularization. The method identifies unknown potential coefficients in elliptic PDEs from noisy measurements.

Paper: [Potential Identification via Tikhonov-PINNs](https://iopscience.iop.org/article/10.1088/1361-6420/ae199a)

## Overview

The codebase implements a dual-network architecture:
- **q_net**: Estimates the unknown potential coefficient $q(x)$
- **u_net**: Approximates the PDE solution $u(x)$

The method solves inverse problems of the form:

$$-\Delta u + q u = f \quad \text{in } \Omega$$

with Neumann boundary conditions:

$$\frac{\partial u}{\partial n} = g \quad \text{on } \partial\Omega$$

## Repository Structure

```
Tikhonov-PINNs/
├── TikPINN/               # Main PINN solver package
│   ├── main.py            # Entry point for training
│   ├── tune_hyperparams.py # Optuna hyperparameter optimization
│   ├── submit_jobs.py     # SLURM batch job submission
│   ├── run.slurm          # SLURM submission script template
│   ├── GenerateData/      # Synthetic data generation
│   │   ├── generate_data*.py   # Data generators for different examples
│   │   └── problems/      # Problem definitions (Example01-06)
│   ├── model/             # Core model components
│   │   ├── nn.py          # MLP network with residual connections
│   │   ├── loss.py        # TikPINN loss (measurement + PDE + regularization)
│   │   ├── data.py        # Datasets and distributed samplers
│   │   ├── optim.py       # Adam/LBFGS optimizers with warmup scheduler
│   │   ├── train.py       # Three-phase training pipeline
│   │   ├── utils.py       # Auto-diff utilities, norms, checkpointing
│   │   └── problem.py     # PDE operators and problem base classes
│   ├── config/            # YAML configuration files
│   └── requirements.txt   # Python dependencies
│
├── TikFEM/                # FEM-based solver (dolfin-adjoint)
│   ├── solvepde.py        # Forward PDE solver
│   ├── gaussian_peak.py   # Gaussian peak test problem
│   ├── non_smooth.py      # Non-smooth coefficient cases
│   └── utils.py           # Mesh generation and utilities
│
├── Visualization/         # Result plotting scripts
│   ├── plot_*.py          # Various visualization scripts
│
└── CLAUDE.md              # Project documentation
```

## Installation

### TikPINN (PyTorch-based)

```bash
cd TikPINN
pip install -r requirements.txt
```

**Requirements:**
- Python >= 3.8
- PyTorch >= 2.0.0
- CUDA-compatible GPU (recommended for training)

### TikFEM (FEniCS-based)

```bash
# Install FEniCS (platform-specific)
# See https://fenicsproject.org/download/

cd TikFEM
pip install -r requirements.txt  # if available
```

## Quick Start

### 1. Generate Training Data

```bash
cd TikPINN/GenerateData

# Example 03: Standard test problem
python generate_data_example03.py

# Example 05: Gaussian peak
python generate_data_example05.py

# All available examples:
# - example01.py: 2D exp(x) potential
# - example02.py: 2D test problem
# - example03.py: Standard problem (1D/2D)
# - example04.py: Width/noise sensitivity study
# - example05.py: Gaussian peak
# - example06.py: 1D (1+x)*exp(x) potential
```

### 2. Configure Training

Edit `TikPINN/config/params.yml`:

```yaml
task:
  idx: "05"           # Problem ID
  noise_str: "01"     # Noise level (01=1%, 10=10%)
  dim: 2              # Problem dimension

dataloader_params:
  batch_size: 2500
  n_samples: 50000

q_net_params:
  width: 128
  depth: 4
  activation: "tanh"
  box: [1, 5]         # Output bounds for q

u_net_params:
  width: 64
  depth: 5
  activation: "swish"
  box: null

loss_params:
  alpha: 1.0          # Measurement loss weight
  lamb: 1.0e-7        # Tikhonov regularization weight
  regularization: "H2"  # Options: H2, L2, 0

optim_params_adam:
  q_lr: 1.0e-4
  u_lr: 1.0e-4

train_params:
  pretrain_epochs_u: 100
  num_epochs: [5000, 0]  # [Adam epochs, LBFGS epochs]
```

### 3. Run Training

**Single GPU/CPU:**
```bash
cd TikPINN
python main.py --config_path config/params.yml
```

**Multi-GPU (DDP):**
```bash
# Automatically uses all available GPUs
python main.py --config_path config/params.yml
```

**Resume from checkpoint:**
```bash
# Place checkpoint.pt in results folder or set checkpoint_path in config
python main.py --config_path config/params.yml
```

### 4. Hyperparameter Tuning

```bash
cd TikPINN
python tune_hyperparams.py --n_trials 50 --config_path config/params.yml
```

### 5. SLURM Cluster Submission

```bash
# Generate batch configs for hyperparameter sweep
python submit_jobs.py --mode noise_sweep
python submit_jobs.py --mode h2_sweep

# Submit jobs
sbatch run.slurm
```

### 6. Visualization

```bash
# View training curves
tensorboard --logdir TikPINN/logs/

# Plot results
cd Visualization
python plot_ex03_fem.py
python plot_heatmap_ex03.py
python plot_training_curves_ex05.py
```

## Methodology

### Loss Function

The total loss combines three components:

$$\mathcal{L}_{total} = \mathcal{L}_{PINNs} + \alpha \mathcal{L}_{measurement} + \lambda \mathcal{L}_{regularization}$$

- **$\mathcal{L}_{PINNs}$**: PDE residual $(-\Delta u + qu - f)^2$ + Neumann BC mismatch
- **$\mathcal{L}_{measurement}$**: MSE between predictions and noisy observations
- **$\mathcal{L}_{regularization}$**: H2 or L2 norm on $q$ (Tikhonov regularization)

### Training Pipeline

The training proceeds in three phases:

1. **Pretraining** (100-1000 epochs): Train u_net on measurement loss only
2. **Joint Training** (1000-5000 epochs): Adam optimizer with warmup + cosine annealing
3. **Fine-tuning** (optional, 0-100 epochs): LBFGS optimizer for final refinement

### Network Architecture

Both networks use MLP with:
- Residual connections (Block structure)
- Configurable depth and width
- Optional output bounding via sigmoid projection
- Xavier initialization

## Data Format

Generated data is stored as `.pt` files:

```python
{
    'int_points': (n_int, dim),      # Interior coordinates
    'bdy_points': (n_bdy, dim),      # Boundary coordinates
    'normal_vec': (n_bdy, dim),      # Outward normals
    'm_int': (n_int, 1),             # Noisy interior measurements
    'm_bdy': (n_bdy, 1),             # Noisy boundary measurements
    'f_val': (n_int, 1),             # Source term values
    'g_val': (n_bdy, 1),             # Boundary flux values
    'u_dagger': (n_int, 1),          # Ground truth solution
    'q_dagger': (n_int, 1),          # Ground truth parameter
}
```

## Problem Index Reference

| idx | Dim | Description |
|-----|-----|-------------|
| 01  | 2D  | exp(x) potential |
| 02  | 2D  | Standard test problem |
| 03  | 1D/2D | Example problem |
| 04  | 1D/2D | Width/noise sensitivity |
| 05  | 2D  | Gaussian peak |
| 06  | 1D  | (1+x)*exp(x) potential |

## TensorBoard Logging

Training metrics are logged to TensorBoard:

- **Loss curves**: total, measurement, PINNs, regularization
- **Error curves**: q_relative_error, u_relative_error
- **Heatmaps**: Predictions vs ground truth (configurable interval)

```yaml
train_params:
  heatmap_every_n_epochs: 1000  # Log heatmaps every N epochs
```

## Checkpoint System

Checkpoints are saved automatically:

```yaml
checkpoint_params:
  save_top_k: 3           # Keep best K models
  save_last: true         # Save final checkpoint
  every_n_epochs: 500     # Save frequency
```

Resume training by placing `checkpoint.pt` in the results directory.

## API Reference

### Core Components

```python
from model.nn import get_network
from model.loss import TikPINNLoss
from model.optim import get_optimizer, get_scheduler
from model.train import train
from model.data import get_dataloader
```

### Loss Components

```python
loss = TikPINNLoss(alpha=1.0, lamb=1e-7, regularization='H2')

# Get all components separately
components = loss.get_loss_components(q_net, u_net, samples)
# Returns: {'total', 'measurement', 'pinns', 'regularization'}
```

### Error Metrics

```python
from model.loss import relative_error_q, relative_error_u

q_rel_err = relative_error_q(q_net, samples)
u_rel_err = relative_error_u(u_net, samples)
```

## Citing

If you use this code in your research, please cite:

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

## Contact

For issues and questions, please open an issue on the repository.
