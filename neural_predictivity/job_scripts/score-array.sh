#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=3:0:0
#SBATCH --mem=32G
#SBATCH --array=10,11,14,15

module load python/3.11 scipy-stack
module load opencv
module load mpi4py
source ~/comp400/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TORCH_HOME='/project/aip-bashivan/mayft/torch_home'
export BRAINIO_HOME="$SCRATCH/brainio"
export RESULTCACHING_HOME=$SCRATCH
cd factorized-neural-predictivity
python model_evaluation.py $SLURM_ARRAY_TASK_ID
