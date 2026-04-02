#!/bin/bash

#SBATCH --account yangzhijian
#SBATCH --job-name TikPINN
#SBATCH --partition gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --output job-%A_%a.out
#SBATCH --error job-%A_%a.err
#SBATCH --array=1-4%4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2

RUN_PATH="."
cd "$RUN_PATH" || exit 1

echo Directory is $PWD
echo This job runs on the following nodes: $SLURM_JOB_NODELIST
echo This job has allocated $SLURM_JOB_CPUS_PER_NODE cpu cores.
echo Problem: one_peak_1d
echo Time is `date`
echo ---------------------------------------------

line=$(sed -n "${SLURM_ARRAY_TASK_ID}p" ${RUN_PATH}/config/args.txt)

read -a ARGS <<< "$line"

source /project/songpengcheng/miniconda3/etc/profile.d/conda.sh
conda activate torch
# Run the program with the parameter file
echo "Running with parameters: ${ARGS[@]}"
/project/songpengcheng/miniconda3/envs/torch/bin/python -u main.py --config=./config/${ARGS[@]}

echo ---------------------------------------------
echo Time is `date`
