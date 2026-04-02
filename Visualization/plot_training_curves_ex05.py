"""
Plot training curves from tensorboard logs for Example 05.

Compares 1% noise (v55) and 10% noise (v54) for:
- q_relative error
- u_relative error
- training loss
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from tensorboard.backend.event_processing import event_accumulator
from scipy.ndimage import uniform_filter1d


def smooth_curve(steps, values, window_size=500):
    """Smooth curve using uniform filter."""
    if len(values) < window_size:
        return steps, values
    smoothed_values = uniform_filter1d(values, size=window_size, mode='nearest')
    return steps, smoothed_values


def load_tensorboard_scalars(log_file):
    """Load scalar data from tensorboard log file."""
    ea = event_accumulator.EventAccumulator(log_file,
        size_guidance={
            event_accumulator.SCALARS: 0,  # Load all scalars
        })
    ea.Reload()

    scalars = {}
    for tag in ea.Tags()['scalars']:
        events = ea.Scalars(tag)
        scalars[tag] = {
            'steps': [e.step for e in events],
            'values': [e.value for e in events],
            'wall_times': [e.wall_time for e in events],
        }
    return scalars


def plot_training_curves(v54_scalars, v55_scalars, output_dir):
    """Plot comparison of training curves between 1% and 10% noise."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    colors = {
        'v54': '#d62728',  # 10% noise - red
        'v55': '#1f77b4',  # 1% noise - blue
    }

    labels = {
        'v54': r'$\delta$=10%',
        'v55': r'$\delta$=1%',
    }

    # Plot 1: Training Loss (joint loss)
    ax = axes[0]
    for version, scalars, label, color in [
        ('v54', v54_scalars, labels['v54'], colors['v54']),
        ('v55', v55_scalars, labels['v55'], colors['v55']),
    ]:
        if 'Loss/joint' in scalars:
            steps = scalars['Loss/joint']['steps']
            values = scalars['Loss/joint']['values']
            # Plot raw data with low alpha
            ax.plot(steps, values, alpha=0.2, color=color, linewidth=0.5)
            # Plot smoothed curve
            steps_smooth, values_smooth = smooth_curve(steps, values, window_size=50)
            ax.plot(steps_smooth, values_smooth, label=label, color=color, linewidth=1.5)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training Loss (Joint)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # Plot 2: Q Relative Error
    ax = axes[1]
    for version, scalars, label, color in [
        ('v54', v54_scalars, labels['v54'], colors['v54']),
        ('v55', v55_scalars, labels['v55'], colors['v55']),
    ]:
        if 'Error/q_relative' in scalars:
            steps = scalars['Error/q_relative']['steps']
            values = scalars['Error/q_relative']['values']
            # Plot raw data with low alpha
            ax.plot(steps, values, alpha=0.2, color=color, linewidth=0.5)
            # Plot smoothed curve
            steps_smooth, values_smooth = smooth_curve(steps, values, window_size=50)
            ax.plot(steps_smooth, values_smooth, label=label, color=color, linewidth=1.5)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Relative Error', fontsize=12)
    ax.set_title(r'$q$ Relative Error', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # Plot 3: U Relative Error
    ax = axes[2]
    for version, scalars, label, color in [
        ('v54', v54_scalars, labels['v54'], colors['v54']),
        ('v55', v55_scalars, labels['v55'], colors['v55']),
    ]:
        if 'Error/u_relative' in scalars:
            steps = scalars['Error/u_relative']['steps']
            values = scalars['Error/u_relative']['values']
            # Plot raw data with low alpha
            ax.plot(steps, values, alpha=0.2, color=color, linewidth=0.5)
            # Plot smoothed curve
            steps_smooth, values_smooth = smooth_curve(steps, values, window_size=50)
            ax.plot(steps_smooth, values_smooth, label=label, color=color, linewidth=1.5)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Relative Error', fontsize=12)
    ax.set_title(r'$u$ Relative Error', fontsize=14)
    ax.set_ylim(0.001, 1)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    fig.tight_layout()

    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'ex05_training_curves.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.show()

    return fig


def main():
    logs_base_dir = './logs/ex05'
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'Figures', 'ex05')

    # v54: 10% noise, v55: 1% noise
    log_files = {
        'v54': os.path.join(logs_base_dir, 'v54', 'events.out.tfevents.1774927785.g0060.600685.0'),
        'v55': os.path.join(logs_base_dir, 'v55', 'events.out.tfevents.1774927785.g0060.600686.0'),
    }

    # Load tensorboard data
    print("Loading tensorboard logs...")
    v54_scalars = load_tensorboard_scalars(log_files['v54'])
    v55_scalars = load_tensorboard_scalars(log_files['v55'])

    print(f"\nv54 (10% noise) available scalars: {list(v54_scalars.keys())}")
    print(f"v55 (1% noise) available scalars: {list(v55_scalars.keys())}")

    # Plot training curves
    plot_training_curves(v54_scalars, v55_scalars, output_dir)

    print(f"\nPlots saved to: {output_dir}")


if __name__ == "__main__":
    main()
