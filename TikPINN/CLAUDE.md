# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TikPINN (Tikhov Physics-Informed Neural Networks) - A PyTorch implementation for solving 1D/nD inverse problems using PINNs with Tikhonov regularization. Supports single-GPU and multi-GPU DDP training.

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
- TensorBoard logging support
- Checkpoint save/load for resuming training

### Model Components (`model/`)
- **`nn.py`**: `MLP` with residual connections, Tanh activation, sigmoid box projection for bounded outputs
- **`loss.py`**: `TikPINNLoss` = measurement_loss + α×PINNS_loss + λ×regularization_loss (L2/H2)
- **`data.py`**: `TikDataset` with .pt/.txt format support, `DistributedSampler` for DDP
- **`optim.py`**: Adam/AdamW/LBFGS optimizers, warmup + cosine annealing scheduler
- **`train.py`**: Three-phase training - pretrain u_net → Adam joint → LBFGS fine-tuning
- **`utils.py`**: Config validation, auto-diff (grad/laplace), H2/L2 norms, seed setting, version management
- **`problem.py`**: PDE operators (`elliptic`, `neumann`) and evaluation helpers

### Data Generation (`GenerateData/`)
- **`generate_data.py`**: Unified 1D/nD data generation with noise
- **`problems/problem_base_1d.py`**: Abstract `Problem1D` base class
- **`problems/problem_base_nd.py`**: Abstract `ProblemND` base class for nD problems
- **`problems/`**: Concrete implementations (`example01.py`, `example02.py`, `example06.py`, `sine_product_nd.py`)

### Configuration (`config/params.py`)
- Helper scripts to batch-generate YAML configs for hyperparameter sweeps (noise, λ, width, seeds, n_samples)

## Data Format

Training data stored as `.pt` files:
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

`TikDataset` pairs interior/boundary samples with independent sampling via modulo indexing.

## Config Structure (`params.yaml`)

```yaml
task:
  idx: "06"        # Problem ID (01=2D, 02=2D, 06=1D)
  noise_str: "01"  # Noise level (01=1%, 10=10%)
  dim: 2           # Problem dimension

dataloader_params:
  data_path: './data'
  batch_size: 2500
  n_samples: 50000

q_net_params/u_net_params:
  in_features: <auto-set>
  width: 64           # Hidden layer width
  depth: 2            # Number of hidden layers
  box: [lower, upper]  # Output bounds (use [] for unbounded)

loss_params:
  alpha: 1.0         # PINNS loss weight
  lamb: 1.0e-7       # Tikhonov regularization weight
  regularization: "H2"  # or "L2", "0"

pretrain_optim_params:
  pretrain_u_lr: 1.0e-4
  pretrain_u_reg: 0.0

optim_params_adam:
  q_lr: 1.0e-4
  u_lr: 1.0e-4
  weight_decay: 0.0

optim_params_lbfgs:
  q_lr: 1.0e-4
  u_lr: 1.0e-4
  line_search_fn: "strong_wolfe"
  max_iter: 20

scheduler_params:
  warmup_steps: 100
  total_steps: 10000  # Adam_epochs + LBFGS_epochs

train_params:
  pretrain_epochs_u: 100
  num_epochs: [1000, 100]  # [Adam_epochs, LBFGS_epochs]
  logs_path: './logs'
  heatmap_every_n_epochs: 100  # Log heatmaps every N epochs (0 to disable)

checkpoint_params:
  save_top_k: 3
  save_last: true
  every_n_epochs: 500

seed: 42
```

## Key Patterns

- **Training phases**:
  1. Pretrain u_net (measurement loss only)
  2. Joint training with Adam (warmup + cosine annealing)
  3. Fine-tuning with LBFGS

- **DDP training**: NCCL backend, `MASTER_ADDR=127.0.0.1:64060`

- **Results versioning**: Auto-incrementing version folders (`logs/ex{idx}/v1/`, `v2/`, ...)

- **TensorBoard**: Logs to `logs/ex{idx}/v{N}/` with scalars for loss and relative errors

- **Checkpointing**: Supports resume from `checkpoint.pt` with optimizer/scheduler state

## Common Operations

```bash
# Run with specific config
python main.py --config_path config/params.yml
```

### Hyperparameter Tuning with Optuna

```bash
# Basic usage (50 trials)
python tune_hyperparams.py --n_trials 50 --config_path config/params.yml

# Parallel tuning (4 parallel jobs)
python tune_hyperparams.py --n_trials 50 --n_jobs 4 --config_path config/params.yml

# Save/load study to database (resume later)
python tune_hyperparams.py --n_trials 50 --storage sqlite:///study.db --study_name my_study
```

**Tunable parameters:**
| Parameter | Range |
|-----------|-------|
| `q_width` | 32-256 (step 32) |
| `q_depth` | 1-4 |
| `u_width` | 32-256 (step 32) |
| `u_depth` | 1-4 |
| `alpha` | 0.1-10.0 (log scale) |
| `lamb` | 1e-10 - 1e-4 (log scale) |
| `q_lr` | 1e-5 - 1e-2 (log scale) |
| `u_lr` | 1e-5 - 1e-2 (log scale) |
| `batch_size` | [1000, 2500, 5000, 10000] |

Best config saved to `config/params_best.yml` after tuning.

## Working Constraints

- **No extra features**: Only implement explicitly requested functionality
- **Preserve function names**: Do not rename existing functions
- **Ask for clarification**: Prompt the user when requirements are unclear
- **Concise output**: Summarize code changes briefly instead of printing full code blocks
