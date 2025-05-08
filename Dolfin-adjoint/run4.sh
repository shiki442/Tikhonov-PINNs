#!/bin/bash
#SBATCH -A yangzhijian
#SBATCH -J 1%
#SBATCH --partition=pub
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH -o job-%j.out
#SBATCH -e job-%j.err

RUN_PATH="."
cd "$RUN_PATH" || exit 1

echo Directory is $PWD
echo This job runs on the following nodes: $SLURM_JOB_NODELIST
echo This job has allocated $SLURM_JOB_CPUS_PER_NODE cpu cores.

echo Problem: discontinuous
echo Time is `date`
echo ---------------------------------------------

source /project/songpengcheng/miniconda3/etc/profile.d/conda.sh
conda activate fenics
/project/songpengcheng/miniconda3/envs/fenics/bin/python -u iter.py

echo ---------------------------------------------
echo End at `date`