# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Tikhonov-PINNs** - A PyTorch implementation for solving inverse problems using Physics-Informed Neural Networks (PINNs) with Tikhonov regularization. The repository contains:

- **TikPINN/** - Main PINN solver for inverse potential identification
- **TikFEM/** - FEM-based solver using dolfin-adjoint for comparison/validation
- **Visualization/** - Plotting scripts for results analysis

## Quick Start

```bash
# Install dependencies (TikPINN)
cd TikPINN
pip install -r requirements.txt

# Generate training data
cd GenerateData
python generate_data_example03.py  # or other example generators

# Run training (single GPU or CPU)
cd ..
python main.py --config_path config/params.yml

# Hyperparameter tuning with Optuna
python tune_hyperparams.py --n_trials 50 --config_path config/params.yml

# Submit batch jobs on SLURM cluster
python submit_jobs.py --mode noise_sweep
sbatch run.slurm
```

## Architecture

### TikPINN Structure

```
TikPINN/
├── main.py              # Entry point: single-GPU and multi-GPU DDP training
├── tune_hyperparams.py  # Optuna hyperparameter optimization
├── submit_jobs.py       # SLURM batch job generator
├── run.slurm            # SLURM submission script
├── GenerateData/        # Synthetic data generation
│   ├── generate_data*.py
│   └── problems/        # Problem definitions (example01-06, base classes)
├── model/
│   ├── nn.py            # MLP with residual connections, box projection
│   ├── loss.py          # TikPINNLoss: measurement + PINNs + regularization
│   ├── data.py          # TikDataset, BatchTikDataset, DistributedSampler
│   ├── optim.py         # Adam/LBFGS optimizers, warmup + cosine scheduler
│   ├── train.py         # 3-phase training: pretrain → Adam → LBFGS
│   ├── utils.py         # Auto-diff, H2/L2 norms, config validation
│   └── problem.py       # PDE operators (elliptic, neumann)
└── config/              # YAML configurations
```

### TikFEM Structure

Uses FEniCS/dolfin-adjoint for FEM-based inverse problem solving:

- **solvepde.py** - Forward PDE solver with Neumann/Dirichlet BCs
- **gaussian_peak.py**, **one_peak.py** - Specific test problems
- **non_smooth.py**, **discontinuous.py** - Non-smooth coefficient cases
- **utils.py** - Mesh generation, result saving helpers

### Core Training Pipeline (TikPINN)

**main.py** handles:
- Config loading from YAML
- DDP setup via `mp.spawn()` with NCCL backend (`MASTER_ADDR=127.0.0.1:64060`)
- Two-network architecture: `q_net` (parameter estimation) + `u_net` (solution)
- Checkpoint save/load for resuming

**Loss Function** (`loss.py`):
```
total = PINNs_loss + α × measurement_loss + λ × regularization_loss
```
- `measurement_loss`: MSE between predictions and noisy observations
- `PINNs_loss`: PDE residual (-Δu + qu = f) + Neumann BC mismatch
- `regularization_loss`: H2 or L2 norm on q

**Training Phases** (`train.py`):
1. Pretrain u_net (measurement loss only, 100-1000 epochs)
2. Joint training with Adam + warmup + cosine annealing (1000-5000 epochs)
3. Fine-tuning with LBFGS (optional, 0-100 epochs)

### Data Format

Generated data stored as `.pt` files:
```python
{
    'int_points': (n_int, dim),      # Interior coordinates
    'bdy_points': (n_bdy, dim),      # Boundary coordinates
    'normal_vec': (n_bdy, dim),      # Outward normals
    'm_int': (n_int, 1),             # Noisy interior measurements
    'm_bdy': (n_bdy, 1),             # Noisy boundary measurements
    'f_val': (n_int, 1),             # Source term (-Δu + q*u)
    'g_val': (n_bdy, 1),             # Boundary flux (∇u·n)
    'u_dagger': (n_int, 1),          # Ground truth solution
    'q_dagger': (n_int, 1),          # Ground truth parameter
}
```

### Configuration Structure

Key YAML sections:
```yaml
task:
  idx: "03"           # Problem ID (01-06)
  noise_str: "01"     # Noise level (00=0%, 01=1%, 10=10%)
  dim: 2              # Problem dimension

dataloader_params:
  batch_size: 2500
  n_samples: 50000

q_net_params/u_net_params:
  width: 64           # Hidden layer width
  depth: 2            # Number of hidden layers
  box: [lower, upper] # Output bounds (null for unbounded)
  activation: tanh|relu6p|swish

loss_params:
  alpha: 1.0          # PINNs loss weight
  lamb: 1e-7          # Tikhonov regularization weight
  regularization: H2|L2|0

optim_params_adam/lbfgs:
  q_lr, u_lr: learning rates

train_params:
  pretrain_epochs_u: 100
  num_epochs: [1000, 100]  # [Adam_epochs, LBFGS_epochs]
  heatmap_every_n_epochs: 100  # TensorBoard heatmap logging

checkpoint_params:
  save_top_k: 3
  every_n_epochs: 500
```

## Dependencies

**TikPINN** (`TikPINN/requirements.txt`):
- torch>=2.0.0
- tensorboard>=2.14.0
- pyyaml>=6.0
- numpy>=1.24.0
- optuna (for hyperparameter tuning)

**TikFEM**:
- FEniCS/dolfin
- dolfin-adjoint
- matplotlib

## Common Operations

```bash
# Run with specific config
cd TikPINN
python main.py --config_path config/params.yml

# Resume from checkpoint
# (set checkpoint_path in config or place checkpoint.pt in results folder)

# View training curves
tensorboard --logdir TikPINN/logs/

# Generate batch configs for hyperparameter sweep
python submit_jobs.py --mode h2_sweep  # or noise_sweep
```

## Problem Index Mapping

| idx | Dimension | Description |
|-----|-----------|-------------|
| 01  | 2D        | exp(x) potential |
| 02  | 2D        | (verified in code) |
| 03  | 1D/2D     | Example problem |
| 04  | 1D/2D     | Width/noise sensitivity study |
| 05  | 2D        | Gaussian peak |
| 06  | 1D        | (1+x)*exp(x) potential |

## Working Constraints

- Only implement explicitly requested functionality
- Preserve function names and APIs
- Ask for clarification when requirements are unclear
- Provide concise summaries of code changes
