# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TikPINN (Tikhonov Physics-Informed Neural Networks) - A PyTorch implementation for solving 1D inverse problems using PINNs with Tikhonov regularization.

## Quick Start

```bash
# Generate training data (1D example)
cd GenerateData
python generate_data_1d.py

# Run training
python main.py --config_path config/params.yaml
```

## Architecture

### Core Training Pipeline (`main.py`)
- Entry point supporting single GPU and multi-GPU DDP training
- Loads config from YAML, spawns processes via `mp.spawn()` for distributed training
- Two-network architecture: `q_net` (parameter estimation) and `u_net` (solution)

### Model Components (`model/`)
- **`nn.py`**: MLP with residual connections, Tanh activation, sigmoid box projection for bounded outputs
- **`loss.py`**: `TikPINNLoss` = measurement_loss + α×PINNS_loss + λ×regularization_loss (L2/H2)
- **`data.py`**: DataLoader with .pt/.txt format support, distributed sampler for DDP
- **`optim.py`**: Adam/AdamW/LBFGS optimizers, warmup + cosine annealing scheduler
- **`train.py`**: Two-phase training - pretrain u_net, then joint training (Adam → LBFGS)
- **`utils.py`**: Config validation, automatic differentiation (grad/laplace), H2/L2 norms, seed setting

### Data Generation (`GenerateData/`)
- **`problem_base.py`**: Abstract `Problem1D` base class defining q_dagger, u_dagger, and derivatives
- **`generate_data_1d.py`**: Generates interior/boundary points with configurable noise
- **`problems/`**: Concrete problem definitions (e.g., `example06.py`)

### Configuration (`config/params.py`)
- Helper scripts to batch-generate YAML configs for hyperparameter sweeps (noise, regularization, network width, seeds)

## Data Format

Training data stored as `.pt` files with keys:
- `int_points`, `bdy_points`, `normal_vec`: Geometry (n, 1)
- `m_int`, `m_bdy`: Noisy measurements
- `f_val`: Source term (-Δu + q*u)
- `g_val`: Boundary flux (∇u·n)
- `u_dagger`, `q_dagger`: Ground truth

## Config Structure (`params.yaml`)

```yaml
task:
  idx: "06"        # Problem ID
  noise_str: "10"  # Noise level suffix
  dim: 1
dataloader_params:
  batch_size: 256
  n_samples: 50000
q_net_params/u_net_params:
  width_list: [32, 32]
  box: [lower, upper]  # Output bounds
loss_params:
  alpha: 1.0       # PINNS weight
  lamb: 1e-8       # Tikhonov regularization
  regularization: "H2"  # or "L2", "0"
train_params:
  pretrain_epochs_u: 1000
  num_epochs: [5000, 5000]  # [Adam_epochs, LBFGS_epochs]
```

## Key Patterns

- **Data layout**: Sample tensor columns = [int_point, bdy_point, normal, m_int, m_bdy, f_val, g_val, u_dagger, q_dagger]
- **Two-phase optimization**: Adam (warmup+cosine) → LBFGS for fine-tuning
- **DDP training**: Uses NCCL backend, MASTER_ADDR=localhost:64060
- **Results**: Saved to `output/{problem_name}/result_{width}_{seed}/`

## Common Operations

```bash
# Generate configs for noise sweep (params.py)
python config/params.py  # Edit bash_cfg_noise() for desired ranges

# Run with specific config
python main.py --config_path config/params_08_10.yaml
```

## Working Constraints

- **No extra features**: Only implement explicitly requested functionality
- **Preserve function names**: Do not rename existing functions
- **Ask for clarification**: Prompt the user when requirements are unclear
- **Concise output**: Summarize code changes briefly instead of printing full code blocks
