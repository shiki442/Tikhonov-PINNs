#!/bin/bash
#SBATCH -A yangzhijian
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH -J TikPINN
#SBATCH -o output.log
#SBATCH -e err.log
#SBATCH -p gpu

PYTHON_PATH=/home/cgduan/project/miniconda3/envs/venv/bin
$PYTHON_PATH/python -u main.py --config=./params.yml