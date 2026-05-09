#!/bin/bash
#SBATCH --nodes 1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --time=8:0:0
#SBATCH --mem=64G

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TORCH_HOME='/project/aip-bashivan/mayft/torch_home'
export BRAINIO_HOME="$SCRATCH/brainio"
export RESULTCACHING_HOME=$SCRATCH
cd factorized-neural-predictivity
python model_scores.py
