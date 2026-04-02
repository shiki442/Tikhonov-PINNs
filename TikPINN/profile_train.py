"""
Performance profiling script for TikPINN training.
Profiles key components: data loading, forward pass, loss computation, backward pass, etc.
"""

import os
import time
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from model.data import get_dataloader
from model.loss import get_loss
from model.nn import get_network
from model.optim import get_optimizer, get_pretrain_optimizer
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'GenerateData'))
from GenerateData.problems import Example03Problem
from model.utils import set_seed


def profile_function(func, *args, warmup=3, iterations=10, device='cuda'):
    """Profile a function and return timing statistics."""
    if device != 'cpu':
        torch.cuda.synchronize(device)

    # Warmup
    for _ in range(warmup):
        func(*args)
        if device != 'cpu':
            torch.cuda.synchronize(device)

    # Timed runs
    times = []
    for _ in range(iterations):
        if device != 'cpu':
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

        func(*args)

        if device != 'cpu':
            end.record()
            torch.cuda.synchronize(device)
            times.append(start.elapsed_time(end))
        else:
            times.append(0)  # Placeholder for CPU timing

    return {
        'mean': sum(times) / len(times),
        'std': torch.std(torch.tensor(times)).item() if len(times) > 1 else 0,
        'min': min(times),
        'max': max(times),
    }


def main():
    # Load config
    config_path = 'config/params.yml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_str)
    set_seed(seed=config.get('seed', 42))

    # Set in_features based on problem dimension
    dim = config['task']['dim']
    config['q_net_params']['in_features'] = dim
    config['u_net_params']['in_features'] = dim

    print("=" * 60)
    print("TikPINN Performance Profiling")
    print("=" * 60)
    print(f"Device: {device_str}")
    print(f"Task: example{config['task']['idx']}, noise={config['task']['noise_str']}")
    print(f"Batch size: {config['dataloader_params']['batch_size']}")
    print(f"n_samples: {config['dataloader_params']['n_samples']}")
    print(f"Problem dimension: {dim}")
    print()

    # ==================== Data Loading Profile ====================
    print("=" * 60)
    print("1. DATA LOADING PROFILE")
    print("=" * 60)

    dataloader = get_dataloader(idx=config['task']['idx'], noise_str=config['task']['noise_str'], **config['dataloader_params'])

    # Profile data iteration
    def iterate_dataloader():
        for batch in dataloader:
            _ = (batch[0].to(device), batch[1].to(device))
            break  # Just one batch

    dl_times = []
    for _ in range(10):
        start = time.time()
        iterate_dataloader()
        dl_times.append(time.time() - start)

    print(
        f"Iterate one batch (mean ± std): {1000*sum(dl_times)/len(dl_times):.2f} ± {1000*torch.std(torch.tensor(dl_times)).item():.2f} ms"
    )
    print()

    # ==================== Network Forward Pass Profile ====================
    print("=" * 60)
    print("2. NETWORK FORWARD PASS PROFILE")
    print("=" * 60)

    q_net = get_network(**config['q_net_params']).to(device)
    u_net = get_network(**config['u_net_params']).to(device)

    # Get input shape
    batch_size = config['dataloader_params']['batch_size']

    # Create dummy input
    int_x = torch.randn(batch_size, dim, device=device)
    bdy_x = torch.randn(batch_size // 2, dim, device=device)

    def q_forward():
        _ = q_net(int_x)

    def u_forward():
        _ = u_net(int_x)

    q_stats = profile_function(q_forward, device=device_str)
    u_stats = profile_function(u_forward, device=device_str)

    print(f"q_net forward ({batch_size} samples): {q_stats['mean']:.2f} ± {q_stats['std']:.2f} ms")
    print(f"u_net forward ({batch_size} samples): {u_stats['mean']:.2f} ± {u_stats['std']:.2f} ms")
    print()

    # ==================== Gradient Computation Profile ====================
    print("=" * 60)
    print("3. GRADIENT COMPUTATION PROFILE (PDE Residual)")
    print("=" * 60)

    from model.problem import elliptic, neumann
    from model.utils import laplace, grad

    # Create sample data
    interior = torch.randn(batch_size, dim, device=device, requires_grad=True)
    f_val = torch.randn(batch_size, 1, device=device)
    bdy = torch.randn(batch_size // 2, dim, device=device, requires_grad=True)
    normal = torch.randn(batch_size // 2, dim, device=device)
    g_val = torch.randn(batch_size // 2, 1, device=device)

    def compute_pde_residual():
        u_net.train()
        u_val = u_net(interior)
        q_val = q_net(interior)
        residual = f_val + laplace(u_net, interior) - q_val * u_val
        return residual

    def compute_neumann_bc():
        u_net.train()
        grad_u = grad(u_net, bdy)
        bc = torch.sum(grad_u * normal, dim=1, keepdim=True)
        return bc

    pde_stats = profile_function(compute_pde_residual, device=device_str)
    neumann_stats = profile_function(compute_neumann_bc, device=device_str)

    print(f"PDE residual (elliptic): {pde_stats['mean']:.2f} ± {pde_stats['std']:.2f} ms")
    print(f"Neumann BC: {neumann_stats['mean']:.2f} ± {neumann_stats['std']:.2f} ms")
    print()

    # ==================== Loss Computation Profile ====================
    print("=" * 60)
    print("4. LOSS COMPUTATION PROFILE")
    print("=" * 60)

    loss_fn = get_loss(**config['loss_params'])

    # Create full samples
    m_int = torch.randn(batch_size, 1, device=device)
    m_bdy = torch.randn(batch_size // 2, 1, device=device)
    u_dagger = torch.randn(batch_size, 1, device=device)
    q_dagger = torch.randn(batch_size, 1, device=device)

    int_sample = torch.cat([interior.detach(), m_int, f_val, u_dagger, q_dagger], dim=1)
    bdy_data = torch.cat([bdy.detach(), normal, m_bdy, g_val], dim=1)

    samples = (int_sample, bdy_data)

    def full_loss():
        q_net.train()
        u_net.train()
        loss = loss_fn(q_net, u_net, samples)
        return loss

    def measurement_loss():
        u_net.train()
        loss = loss_fn.measurement(u_net, samples)
        return loss

    def get_loss_components():
        q_net.train()
        u_net.train()
        components = loss_fn.get_loss_components(q_net, u_net, samples)
        return components

    full_loss_stats = profile_function(full_loss, device=device_str)
    meas_loss_stats = profile_function(measurement_loss, device=device_str)
    components_stats = profile_function(get_loss_components, device=device_str)

    print(f"Full loss (PINNs + meas + reg): {full_loss_stats['mean']:.2f} ± {full_loss_stats['std']:.2f} ms")
    print(f"Measurement loss only: {meas_loss_stats['mean']:.2f} ± {meas_loss_stats['std']:.2f} ms")
    print(f"Loss components: {components_stats['mean']:.2f} ± {components_stats['std']:.2f} ms")
    print()

    # ==================== Backward Pass Profile ====================
    print("=" * 60)
    print("5. BACKWARD PASS PROFILE")
    print("=" * 60)

    def forward_backward():
        q_net.train()
        u_net.train()
        loss = loss_fn(q_net, u_net, samples)
        loss.backward()
        return loss

    def forward_backward_lbfgs():
        q_net.train()
        u_net.train()

        def closure():
            q_net.zero_grad()
            u_net.zero_grad()
            loss = loss_fn(q_net, u_net, samples)
            loss.backward()
            return loss

        return closure

    fb_stats = profile_function(forward_backward, device=device_str)

    # Profile LBFGS closure (multiple calls)
    lbfgs_times = []
    for _ in range(10):
        optimizer = torch.optim.LBFGS(
            list(q_net.parameters()) + list(u_net.parameters()),
            lr=1e-4,
            max_iter=5,  # Use 5 iterations for profiling
            line_search_fn='strong_wolfe' if device_str == 'cpu' else None,
        )

        if device_str != 'cpu':
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

        optimizer.step(forward_backward_lbfgs())

        if device_str != 'cpu':
            end.record()
            torch.cuda.synchronize(device)
            lbfgs_times.append(start.elapsed_time(end))
        else:
            # CPU timing
            pass

    print(f"Forward + Backward (single): {fb_stats['mean']:.2f} ± {fb_stats['std']:.2f} ms")
    if lbfgs_times:
        print(
            f"LBFGS step (5 line search iters): {sum(lbfgs_times)/len(lbfgs_times):.2f} ± {torch.std(torch.tensor(lbfgs_times)).item():.2f} ms"
        )
    print()

    # ==================== Validation Profile ====================
    print("=" * 60)
    print("6. VALIDATION PROFILE")
    print("=" * 60)

    from model.train import _compute_errors_on_grid

    problem = Example03Problem()

    def validate_on_grid():
        q_net.eval()
        u_net.eval()
        q_err, u_err = _compute_errors_on_grid(device, q_net, u_net, problem, eval_points=101)
        return q_err, u_err

    val_stats = profile_function(validate_on_grid, device=device_str)
    print(f"Validation on 101x101 grid: {val_stats['mean']:.2f} ± {val_stats['std']:.2f} ms")
    print()

    # ==================== Full Epoch Simulation ====================
    print("=" * 60)
    print("7. SIMULATED EPOCH TIME (10 batches)")
    print("=" * 60)

    # Adam epoch simulation
    optimizer_adam = torch.optim.Adam([{'params': q_net.parameters(), 'lr': 1e-4}, {'params': u_net.parameters(), 'lr': 1e-4}])

    def run_adam_epoch(n_batches=10):
        q_net.train()
        u_net.train()
        for i in range(n_batches):
            # Create new batch each iteration
            int_x = torch.randn(batch_size, dim, device=device)
            bdy_x = torch.randn(batch_size // 2, dim, device=device)
            m_int = torch.randn(batch_size, 1, device=device)
            m_bdy = torch.randn(batch_size // 2, 1, device=device)
            f_val = torch.randn(batch_size, 1, device=device)
            g_val = torch.randn(batch_size // 2, 1, device=device)
            normal = torch.randn(batch_size // 2, dim, device=device)
            u_dagger = torch.randn(batch_size, 1, device=device)
            q_dagger = torch.randn(batch_size, 1, device=device)

            int_sample = torch.cat([int_x, m_int, f_val, u_dagger, q_dagger], dim=1)
            bdy_sample = torch.cat([bdy_x, normal, m_bdy, g_val], dim=1)
            samples = (int_sample, bdy_sample)

            loss = loss_fn(q_net, u_net, samples)
            optimizer_adam.zero_grad()
            loss.backward()
            optimizer_adam.step()

    # LBFGS epoch simulation
    optimizer_lbfgs = torch.optim.LBFGS(
        list(q_net.parameters()) + list(u_net.parameters()),
        lr=1e-4,
        max_iter=5,
        line_search_fn='strong_wolfe' if device_str == 'cpu' else None,
    )

    def run_lbfgs_epoch(n_batches=10):
        q_net.train()
        u_net.train()
        for i in range(n_batches):
            int_x = torch.randn(batch_size, dim, device=device)
            bdy_x = torch.randn(batch_size // 2, dim, device=device)
            m_int = torch.randn(batch_size, 1, device=device)
            m_bdy = torch.randn(batch_size // 2, 1, device=device)
            f_val = torch.randn(batch_size, 1, device=device)
            g_val = torch.randn(batch_size // 2, 1, device=device)
            normal = torch.randn(batch_size // 2, dim, device=device)
            u_dagger = torch.randn(batch_size, 1, device=device)
            q_dagger = torch.randn(batch_size, 1, device=device)

            int_sample = torch.cat([int_x, m_int, f_val, u_dagger, q_dagger], dim=1)
            bdy_sample = torch.cat([bdy_x, normal, m_bdy, g_val], dim=1)
            samples = (int_sample, bdy_sample)

            def closure():
                loss = loss_fn(q_net, u_net, samples)
                optimizer_lbfgs.zero_grad()
                loss.backward()
                return loss

            optimizer_lbfgs.step(closure)

    # Time Adam epoch
    adam_times = []
    for _ in range(3):
        start = time.time()
        run_adam_epoch(10)
        adam_times.append(time.time() - start)

    # Time LBFGS epoch
    lbfgs_epoch_times = []
    for _ in range(3):
        start = time.time()
        run_lbfgs_epoch(10)
        lbfgs_epoch_times.append(time.time() - start)

    print(f"Adam epoch (10 batches): {sum(adam_times)/len(adam_times):.2f} ± {torch.std(torch.tensor(adam_times)).item():.2f} s")
    print(
        f"LBFGS epoch (10 batches, 5 iters): {sum(lbfgs_epoch_times)/len(lbfgs_epoch_times):.2f} ± {torch.std(torch.tensor(lbfgs_epoch_times)).item():.2f} s"
    )
    print(f"LBFGS is ~{sum(lbfgs_epoch_times)/sum(adam_times):.1f}x slower than Adam")
    print()

    # ==================== Summary ====================
    print("=" * 60)
    print("SUMMARY - Key Bottlenecks")
    print("=" * 60)
    print("1. LBFGS optimizer: Multiple forward/backward per step (line search)")
    print("2. H2 regularization: Requires second-order derivatives")
    print("3. Validation every epoch: Full dataset pass + grid evaluation")
    print("4. Heatmap logging: 101x101 grid inference + matplotlib rendering")
    print()
    print("Recommended optimizations:")
    print("  - Reduce LBFGS max_iter or use Adam only")
    print("  - Use L2 regularization instead of H2")
    print("  - Increase validation interval (e.g., every 10 epochs)")
    print("  - Disable heatmaps or reduce frequency")


if __name__ == '__main__':
    main()
