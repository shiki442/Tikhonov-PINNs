#!/usr/bin/env python3
"""
批量生成 SLURM 任务配置并提交

使用示例:
    python submit_jobs.py --n_trials 20 --partition gpu
"""

import os
import yaml
import subprocess
import argparse
from datetime import datetime

# 默认配置模板
TEMPLATE = {
    'task': {
        'idx': '05',
        'noise_str': '01',
        'dim': 2,
    },
    'dataloader_params': {
        'data_path': './data',
        'batch_size': 2500,
        'n_samples': 50000,
    },
    'q_net_params': {
        'width': 192,
        'depth': 2,
        'activation': 'tanh',
        'box': [1, 5],
    },
    'u_net_params': {
        'width': 64,
        'depth': 5,
        'activation': 'swish',
        'box': None,
    },
    'loss_params': {
        'alpha': 2.5940969818204427,
        'lamb': 1.0549442354773734e-09,
        'regularization': 'H2',
    },
    'pretrain_optim_params': {
        'pretrain_u_lr': 1e-3,
        'pretrain_u_reg': 0.0,
    },
    'optim_params_adam': {
        'name': 'adam',
        'q_lr': 0.0025146444860712847,
        'u_lr': 0.00026278592195837734,
        'weight_decay': 0.0,
    },
    'optim_params_lbfgs': {
        'name': 'lbfgs',
        'q_lr': 1e-4,
        'u_lr': 1e-4,
        'weight_decay': 0.0,
        'line_search_fn': 'strong_wolfe',
        'max_iter': 20,
    },
    'scheduler_params': {
        'warmup_steps': 100,
        'total_steps': 50000,
        'cosine_annealing': True,
    },
    'train_params': {
        'pretrain_epochs_u': 1000,
        'num_epochs': [50000, 0],
        'logs_path': './logs',
        'heatmap_every_n_epochs': 2500,
    },
    'checkpoint_params': {
        'save_top_k': 3,
        'save_last': True,
        'every_n_epochs': 500,
    },
    'seed': 42,
}


def generate_configs(output_dir='config', prefix='params_ex03'):
    """生成多个不同超参数的配置文件"""
    os.makedirs(output_dir, exist_ok=True)

    config_files = []
    
    # 示例 1: 不同的噪声水平
    noise_strs = ['00','01','10','50']

    idx = 0
    for noise in noise_strs:
        idx += 1
        config = TEMPLATE.copy()
        config['task']['noise_str'] = noise
        config['seed'] = 42 + idx  # 不同种子

        filename = f'{prefix}_{idx:03d}.yml'
        filepath = os.path.join(output_dir, filename)

        with open(filepath, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        config_files.append(filename)
        print(f"Generated: {filename} (noise={noise})")

    return config_files


def generate_h2_sweep(output_dir='config', prefix='params_h2'):
    """生成不同正则化参数的配置"""
    os.makedirs(output_dir, exist_ok=True)

    config_files = []

    # 不同的 lambda 值
    lambs = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
    alphas = [0.1, 1.0, 10.0]

    idx = 0
    for lamb in lambs:
        for alpha in alphas:
            idx += 1
            config = TEMPLATE.copy()
            config['loss_params']['lamb'] = lamb
            config['loss_params']['alpha'] = alpha
            config['seed'] = 42 + idx

            filename = f'{prefix}_{idx:03d}.yml'
            filepath = os.path.join(output_dir, filename)

            with open(filepath, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)

            config_files.append(filename)
            print(f"Generated: {filename} (alpha={alpha}, lamb={lamb})")

    return config_files


def create_args_txt(config_files, output_path='config/args.txt'):
    """创建 args.txt 文件"""
    with open(output_path, 'w') as f:
        for cfg in config_files:
            f.write(f"{cfg}\n")
    print(f"Created {output_path} with {len(config_files)} tasks")


def update_slurm_array(num_tasks, max_parallel=4, output_path='run.slurm'):
    """更新 slurm 脚本中的任务数组配置"""
    with open(output_path, 'r') as f:
        content = f.read()

    # 替换 array 配置
    old_line = f"#SBATCH --array=1-"
    new_line = f"#SBATCH --array=1-{num_tasks}%{max_parallel}"

    for line in content.split('\n'):
        if line.startswith('#SBATCH --array='):
            content = content.replace(line, new_line)
            break

    with open(output_path, 'w') as f:
        f.write(content)

    print(f"Updated {output_path}: array=1-{num_tasks}%{max_parallel}")


def submit_jobs(slurm_script='run.slurm', dry_run=False):
    """提交 SLURM 任务"""
    if dry_run:
        print(f"[DRY RUN] Would run: sbatch {slurm_script}")
        return

    result = subprocess.run(['sbatch', slurm_script], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"Submitted: {result.stdout.strip()}")
    else:
        print(f"Error: {result.stderr.strip()}")


def main():
    parser = argparse.ArgumentParser(description='批量提交 SLURM 任务')
    parser.add_argument('--mode', type=str, default='noise_sweep',
                        choices=['lr_sweep', 'h2_sweep', 'custom'],
                        help='配置生成模式')
    parser.add_argument('--n_tasks', type=int, default=None,
                        help='任务数量 (None 则自动生成所有组合)')
    parser.add_argument('--max_parallel', type=int, default=8,
                        help='同时运行的最大任务数')
    parser.add_argument('--dry_run', action='store_true',
                        help='只生成配置，不提交')
    parser.add_argument('--config_dir', type=str, default='config',
                        help='配置文件输出目录')

    args = parser.parse_args()

    print("=" * 60)
    print("SLURM 批量任务提交工具")
    print("=" * 60)

    # 生成配置文件
    if args.mode == 'noise_sweep':
        config_files = generate_configs(output_dir=args.config_dir, prefix='params_ex05')
    elif args.mode == 'h2_sweep':
        config_files = generate_h2_sweep(output_dir=args.config_dir)
    else:
        print("请使用 custom 模式并手动指定配置文件")
        return

    # 限制任务数量
    if args.n_tasks:
        config_files = config_files[:args.n_tasks]

    # 创建 args.txt
    create_args_txt(config_files, output_path=f'{args.config_dir}/args.txt')

    # 更新 slurm 脚本
    update_slurm_array(len(config_files), args.max_parallel,
                       output_path='run.slurm')

    # 提交任务
    print()
    # submit_jobs(dry_run=args.dry_run)

    print()
    print("=" * 60)
    print(f"共生成 {len(config_files)} 个任务配置")
    print(f"任务列表：{args.config_dir}/args.txt")
    print(f"运行命令：sbatch run.slurm")
    print("=" * 60)


if __name__ == '__main__':
    main()
