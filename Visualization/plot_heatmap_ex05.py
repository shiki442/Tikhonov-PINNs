"""
Plot heatmap comparison for Example 05.

Layout: 3 columns (True solution, 1% noise, 10% noise) x 2 rows (q and u)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import torch
import yaml
import sys

# Add parent directory to path to import project modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'TikPINN'))

from model.nn import MLP, get_activation
from GenerateData.problems.example05 import Example05Problem


def load_checkpoint(checkpoint_path, device='cpu'):
    """Load checkpoint and return model state dicts."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    return checkpoint['q_net'], checkpoint['u_net']


def create_models(config):
    """Create q_net and u_net models with config."""
    q_act = get_activation(config['q_net_params']['activation'])
    u_act = get_activation(config['u_net_params']['activation'])

    q_net = MLP(
        in_features=2,
        out_features=1,
        width=config['q_net_params']['width'],
        depth=config['q_net_params']['depth'],
        activation=q_act,
        box=config['q_net_params']['box']
    )
    u_net = MLP(
        in_features=2,
        out_features=1,
        width=config['u_net_params']['width'],
        depth=config['u_net_params']['depth'],
        activation=u_act,
        box=config['u_net_params']['box']
    )
    return q_net, u_net


def get_predictions_on_grid(q_net, u_net, device, eval_points=101):
    """Get predictions and ground truth on a regular grid."""
    q_net.eval()
    u_net.eval()

    # Create 2D grid
    x1 = torch.linspace(0, 1, eval_points, device=device)
    x2 = torch.linspace(0, 1, eval_points, device=device)
    xx1, xx2 = torch.meshgrid(x1, x2, indexing='ij')
    x_grid = torch.stack([xx1.ravel(), xx2.ravel()], dim=1)

    # Compute predictions
    with torch.no_grad():
        q_pred = q_net(x_grid)
        u_pred = u_net(x_grid)

    # Get true solutions
    problem = Example05Problem()
    q_true_np = problem.q_dagger(x_grid.cpu().numpy())
    u_true_np = problem.u_dagger(x_grid.cpu().numpy())

    return {
        'q_pred': q_pred.cpu().numpy().reshape(eval_points, eval_points),
        'u_pred': u_pred.cpu().numpy().reshape(eval_points, eval_points),
        'q_true': q_true_np.reshape(eval_points, eval_points),
        'u_true': u_true_np.reshape(eval_points, eval_points),
        'x1': xx1.cpu().numpy(),
        'x2': xx2.cpu().numpy(),
    }


def plot_heatmaps(data, titles_q, titles_u, filename=None, vmin_q=None, vmax_q=None, vmin_u=None, vmax_u=None):
    """Plot q and u heatmaps in subplots.

    Layout: 3 columns (True, 1% noise, 10% noise) x 2 rows (q, u)
    """
    num_cols = 3  # True, 1% noise, 10% noise
    fig = plt.figure(figsize=(4 * num_cols, 4 * 2))
    gs = gridspec.GridSpec(2, num_cols + 1, width_ratios=[1, 1, 1, 0.05])

    # Define extent for correct coordinate mapping: [xmin, xmax, ymin, ymax]
    extent = [0, 1, 0, 1]

    # Plot q (row 0)
    imgs_q = []
    for i in range(num_cols):
        ax = fig.add_subplot(gs[0, i])
        if i == 0:
            img = ax.imshow(data['q_true'], cmap='coolwarm', aspect='auto', vmin=vmin_q, vmax=vmax_q,
                           origin='lower', extent=extent)
            ax.set_title(titles_q[i], fontsize=18, pad=10)
        else:
            img = ax.imshow(data['q_pred_list'][i-1], cmap='coolwarm', aspect='auto', vmin=vmin_q, vmax=vmax_q,
                           origin='lower', extent=extent)
            ax.set_title(titles_q[i], fontsize=18, pad=10)
        ax.set_xticks([])
        ax.set_yticks([])
        imgs_q.append(img)

    # Colorbar for q
    cbar_ax = fig.add_subplot(gs[0, -1])
    cbar = fig.colorbar(imgs_q[0], cax=cbar_ax)
    cbar.ax.tick_params(labelsize=12)

    # Plot u (row 1)
    imgs_u = []
    for i in range(num_cols):
        ax = fig.add_subplot(gs[1, i])
        if i == 0:
            img = ax.imshow(data['u_true'], cmap='coolwarm', aspect='auto', vmin=vmin_u, vmax=vmax_u,
                           origin='lower', extent=extent)
            ax.set_title(titles_u[i], fontsize=18, pad=10)
        else:
            img = ax.imshow(data['u_pred_list'][i-1], cmap='coolwarm', aspect='auto', vmin=vmin_u, vmax=vmax_u,
                           origin='lower', extent=extent)
            ax.set_title(titles_u[i], fontsize=18, pad=10)
        ax.set_xticks([])
        ax.set_yticks([])
        imgs_u.append(img)

    # Colorbar for u
    cbar_ax = fig.add_subplot(gs[1, -1])
    cbar = fig.colorbar(imgs_u[0], cax=cbar_ax)
    cbar.ax.tick_params(labelsize=12)

    fig.subplots_adjust(left=0.02, right=0.95, top=0.9, bottom=0.02, hspace=0.15, wspace=0.1)

    if filename:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, format='png', dpi=150, bbox_inches='tight')
        print(f"Saved: {filename}")
    plt.show()


def load_config(version_dir):
    """Load config.yaml from version directory."""
    config_path = os.path.join(version_dir, 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_noise_level(version_dir):
    """Get noise level from config.yaml in version directory."""
    config_path = os.path.join(version_dir, 'config.yaml')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        noise_str = config.get('task', {}).get('noise_str', 'unknown')
        return f'{int(noise_str)}%' if noise_str != 'unknown' else 'unknown'
    return 'unknown'


def get_checkpoint_epochs(log_dir):
    """Get list of checkpoint epochs from directory."""
    import re
    pattern = re.compile(r'checkpoint_(\d+)\.pt')
    epochs = []
    for f in os.listdir(log_dir):
        match = pattern.match(f)
        if match:
            epochs.append(int(match.group(1)))
    return sorted(epochs)


def main():
    logs_base_dir = './logs/ex05'
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'Figures', 'ex05')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Map noise levels to versions
    # v55: 1% noise, v54: 10% noise (from config.yaml)
    version_map = {
        '01': 'v55',  # 1% noise
        '10': 'v54',  # 10% noise
    }

    device = 'cpu'
    eval_points = 201

    data = {
        'q_pred_list': [],
        'u_pred_list': [],
    }

    titles_q = [r'$q^{\dagger}$ (True)']
    titles_u = [r'$u^{\dagger}$ (True)']

    # First, compute the ground truth
    problem = Example05Problem()
    x1 = np.linspace(0, 1, eval_points)
    x2 = np.linspace(0, 1, eval_points)
    xx1, xx2 = np.meshgrid(x1, x2)
    x_grid = np.stack([xx1.ravel(), xx2.ravel()], axis=1)

    data['q_true'] = problem.q_dagger(x_grid).reshape(eval_points, eval_points)
    data['u_true'] = problem.u_dagger(x_grid).reshape(eval_points, eval_points)

    # Load predictions from each version
    for noise_str, version in [('01', 'v55'), ('10', 'v54')]:
        version_dir = os.path.join(logs_base_dir, version)

        if not os.path.exists(version_dir):
            print(f"Version directory not found: {version_dir}")
            continue

        # Load config
        config = load_config(version_dir)

        # Get checkpoint epochs
        checkpoint_epochs_avail = get_checkpoint_epochs(version_dir)
        print(f"Available checkpoints in {version}: {checkpoint_epochs_avail[-5:] if len(checkpoint_epochs_avail) > 5 else checkpoint_epochs_avail}")

        if not checkpoint_epochs_avail:
            print(f"No checkpoints found in {version}")
            continue

        # Use final checkpoint
        final_checkpoint = 'checkpoint_final.pt'
        checkpoint_path = os.path.join(version_dir, final_checkpoint)

        if not os.path.exists(checkpoint_path):
            # Fallback to the last available checkpoint
            final_epoch = checkpoint_epochs_avail[-1]
            final_checkpoint = f'checkpoint_{final_epoch}.pt'
            checkpoint_path = os.path.join(version_dir, final_checkpoint)

        q_net, u_net = create_models(config)
        q_net.to(device)
        u_net.to(device)

        q_state, u_state = load_checkpoint(checkpoint_path, device)
        q_net.load_state_dict(q_state)
        u_net.load_state_dict(u_state)

        preds = get_predictions_on_grid(q_net, u_net, device, eval_points=eval_points)
        data['q_pred_list'].append(preds['q_pred'])
        data['u_pred_list'].append(preds['u_pred'])

        titles_q.append(f'$\delta={{{int(noise_str)}\\%}}$')
        titles_u.append(f'$\delta={{{int(noise_str)}\\%}}$')

        print(f"Loaded {version}: noise {noise_str}%, checkpoint: {final_checkpoint}")

    # Determine color scale limits based on true values
    vmin_q = data['q_true'].min()
    vmax_q = data['q_true'].max()
    vmin_u = data['u_true'].min()
    vmax_u = data['u_true'].max()

    # Plot heatmaps
    plot_heatmaps(data, titles_q, titles_u,
                  filename=os.path.join(output_dir, 'ex05_heatmap.png'),
                  vmin_q=vmin_q, vmax_q=vmax_q,
                  vmin_u=vmin_u, vmax_u=vmax_u)

    print(f"\nPlots saved to: {output_dir}")


if __name__ == "__main__":
    main()
