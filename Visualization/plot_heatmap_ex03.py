import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import torch
import yaml

# Add parent directory to path to import project modules
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'TikPINN'))

from model.nn import MLP, get_activation
from GenerateData.problems.example03 import Example03Problem


def load_checkpoint(checkpoint_path, device='cpu'):
    """Load checkpoint and return model state dicts."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    return checkpoint['q_net'], checkpoint['u_net']


def create_models():
    """Create q_net and u_net models with config from ex03v1."""
    q_act = get_activation('tanh')
    u_act = get_activation('swish')

    q_net = MLP(
        in_features=2,
        out_features=1,
        width=128,
        depth=4,
        activation=q_act,
        box=[0, 5]
    )
    u_net = MLP(
        in_features=2,
        out_features=1,
        width=64,
        depth=5,
        activation=u_act,
        box=None
    )
    return q_net, u_net


def compute_relative_error(pred, true):
    """Compute relative L2 error."""
    pred_flat = pred.ravel()
    true_flat = true.ravel()
    numerator = np.sqrt(np.sum((pred_flat - true_flat) ** 2))
    denominator = np.sqrt(np.sum(true_flat ** 2))
    if denominator < 1e-10:
        return np.inf
    return numerator / denominator


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
    problem = Example03Problem()
    q_true_np = problem.q_dagger(x_grid.cpu().numpy())
    u_true_np = problem.u_dagger(x_grid.cpu().numpy())

    # Compute relative errors
    q_pred_np = q_pred.cpu().numpy()
    u_pred_np = u_pred.cpu().numpy()
    q_rel_err = compute_relative_error(q_pred_np, q_true_np)
    u_rel_err = compute_relative_error(u_pred_np, u_true_np)

    return {
        'q_pred': q_pred_np.reshape(eval_points, eval_points),
        'u_pred': u_pred_np.reshape(eval_points, eval_points),
        'q_true': q_true_np.reshape(eval_points, eval_points),
        'u_true': u_true_np.reshape(eval_points, eval_points),
        'x1': xx1.cpu().numpy(),
        'x2': xx2.cpu().numpy(),
        'q_rel_err': q_rel_err,
        'u_rel_err': u_rel_err,
    }


def plot_subplots(data, titles_q, titles_u, filename=None, vmin_q=None, vmax_q=None, vmin_u=None, vmax_u=None):
    """Plot q and u heatmaps in subplots."""
    num_plots = 5
    height = 4
    fig = plt.figure(figsize=(4 * num_plots, 2 * height))
    gs = gridspec.GridSpec(2, 7, width_ratios=[1, 0.3, 1, 1, 1, 1, 0.05])
    imgs = []

    # Plot q (row 0)
    ax = fig.add_subplot(gs[0, 0])
    img = ax.imshow(data['q_true'], cmap='coolwarm', aspect='auto', vmin=vmin_q, vmax=vmax_q)
    ax.invert_xaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(titles_q[0], fontsize=24)

    for i in range(4):
        ax = fig.add_subplot(gs[0, i+2])
        img = ax.imshow(data['q_pred_list'][i], cmap='coolwarm', aspect='auto', vmin=vmin_q, vmax=vmax_q)
        ax.invert_xaxis()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(titles_q[i+1], fontsize=24)
        imgs.append(img)

    cbar_ax = fig.add_subplot(gs[0, -1])
    cbar = fig.colorbar(imgs[0], cax=cbar_ax)
    cbar.ax.tick_params(labelsize=12)

    # Plot u (row 1)
    imgs = []
    ax = fig.add_subplot(gs[1, 0])
    img = ax.imshow(data['u_true'], cmap='coolwarm', aspect='auto', vmin=vmin_u, vmax=vmax_u)
    ax.invert_xaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(titles_u[0], fontsize=24)

    for i in range(4):
        ax = fig.add_subplot(gs[1, i+2])
        img = ax.imshow(data['u_pred_list'][i], cmap='coolwarm', aspect='auto', vmin=vmin_u, vmax=vmax_u)
        ax.invert_xaxis()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(titles_u[i+1], fontsize=24)
        imgs.append(img)

    cbar_ax = fig.add_subplot(gs[1, -1])
    cbar = fig.colorbar(imgs[0], cax=cbar_ax)
    cbar.ax.tick_params(labelsize=12)

    fig.subplots_adjust(left=0.02, right=0.95, top=0.85, bottom=0.02, hspace=0.2, wspace=0.1)

    if filename:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, format='png', dpi=150, bbox_inches='tight')
        print(f"Saved: {filename}")
    plt.show()


def plot_training_curves(metrics_list, labels, filename=None):
    """Plot training loss curves."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))

    for i, (metrics, label) in enumerate(zip(metrics_list, labels)):
        ax = axes[0]
        if 'loss' in metrics:
            epochs = list(metrics['loss'].keys())
            losses = list(metrics['loss'].values())
            ax.plot(epochs, losses, label=label, color=colors[i])
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        if 'q_rel_err' in metrics:
            epochs = list(metrics['q_rel_err'].keys())
            q_errs = list(metrics['q_rel_err'].values())
            ax.plot(epochs, q_errs, label=label, color=colors[i])
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Q Relative Error')
        ax.set_title('Q Relative Error')
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[2]
        if 'u_rel_err' in metrics:
            epochs = list(metrics['u_rel_err'].keys())
            u_errs = list(metrics['u_rel_err'].values())
            ax.plot(epochs, u_errs, label=label, color=colors[i])
        ax.set_xlabel('Epoch')
        ax.set_ylabel('U Relative Error')
        ax.set_title('U Relative Error')
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    if filename:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, format='png', dpi=150, bbox_inches='tight')
        print(f"Saved: {filename}")
    plt.show()


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


def get_noise_level(version_dir):
    """Get noise level from config.yaml in version directory."""
    config_path = os.path.join(version_dir, 'config.yaml')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        noise_str = config.get('task', {}).get('noise_str', 'unknown')
        return f'{int(noise_str)}%' if noise_str != 'unknown' else 'unknown'
    return 'unknown'


def main():
    logs_base_dir = os.path.join(os.path.dirname(__file__), '..', 'TikPINN', 'logs', 'ex03v2')
    output_dir = os.path.join(os.path.dirname(__file__), 'Figures', 'ex03v2')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # List available versions
    versions = [d for d in os.listdir(logs_base_dir) if os.path.isdir(os.path.join(logs_base_dir, d))]
    print(f"Available versions: {versions}")

    # Select versions to plot (default: all)
    versions_to_plot = ['v11', 'v10', 'v6', 'v9']
    versions_to_plot = [v for v in versions_to_plot if v in versions]
    ckpt_to_plot = {'v11':'51000', 'v10':'51000', 'v6':'20000', 'v9':'20000'}

    if not versions_to_plot:
        print("No versions found to plot!")
        return

    device = 'cpu'

    # Select checkpoints to visualize (e.g., 5000, 10000, 15000, 20000)
    # Or use final checkpoint
    checkpoint_epochs = [5000, 10000, 15000, 20000]

    data = {
        'q_pred_list': [],
        'u_pred_list': [],
    }

    # Use first version to get ground truth
    first_version = versions_to_plot[0]
    first_version_dir = os.path.join(logs_base_dir, first_version)

    # Get final checkpoint
    checkpoint_epochs_avail = get_checkpoint_epochs(first_version_dir)
    print(f"Available checkpoints in {first_version}: {checkpoint_epochs_avail}")

    # Use final checkpoint for visualization
    final_checkpoint = f'checkpoint_{checkpoint_epochs[1]}.pt'

    q_net, u_net = create_models()
    q_net.to(device)
    u_net.to(device)

    # Load final checkpoint from first version
    q_state, u_state = load_checkpoint(os.path.join(first_version_dir, final_checkpoint), device)
    q_net.load_state_dict(q_state)
    u_net.load_state_dict(u_state)

    preds = get_predictions_on_grid(q_net, u_net, device, eval_points=101)
    data['q_true'] = preds['q_true']
    data['u_true'] = preds['u_true']

    # Load predictions from each version
    titles_q = [r'$q^{\dagger}$']
    titles_u = [r'$u^{\dagger}$']

    for version in versions_to_plot:
        version_dir = os.path.join(logs_base_dir, version)
        checkpoint_epochs_avail = get_checkpoint_epochs(version_dir)

        if not checkpoint_epochs_avail:
            print(f"No checkpoints found in {version}")
            continue

        final_checkpoint = f'checkpoint_{ckpt_to_plot[version]}.pt'
        checkpoint_path = os.path.join(version_dir, final_checkpoint)

        q_net, u_net = create_models()
        q_net.to(device)
        u_net.to(device)

        q_state, u_state = load_checkpoint(checkpoint_path, device)
        q_net.load_state_dict(q_state)
        u_net.load_state_dict(u_state)

        preds = get_predictions_on_grid(q_net, u_net, device, eval_points=101)
        data['q_pred_list'].append(preds['q_pred'])
        data['u_pred_list'].append(preds['u_pred'])

        noise_level = get_noise_level(version_dir)
        titles_q.append(r'$\delta$'+f'={noise_level}')
        titles_u.append(r'$\delta$'+f'={noise_level}')

        print(f"Loaded {version}: epoch {ckpt_to_plot[version]}, q_rel_err = {preds['q_rel_err']:.4e}, u_rel_err = {preds['u_rel_err']:.4e}")

    # Determine color scale limits
    vmin_q = data['q_true'].min()
    vmax_q = data['q_true'].max()
    vmin_u = data['u_true'].min()
    vmax_u = data['u_true'].max()

    # Combined plot
    plot_subplots(data, titles_q, titles_u, filename=os.path.join(output_dir, 'ex03_unn.png'),
                  vmin_q=vmin_q, vmax_q=vmax_q, vmin_u=vmin_u, vmax_u=vmax_u)

    print(f"\nPlots saved to: {output_dir}")


if __name__ == "__main__":
    main()
