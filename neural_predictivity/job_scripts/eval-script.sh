#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=3:0:0
#SBATCH --mem=50G
#SBATCH --mail-user=mayafthompson@gmail.com

module load python/3.11 scipy-stack
module load opencv
module load mpi4py
source ~/comp400/bin/activate
export TORCH_HOME='/project/aip-bashivan/mayft/torch_home'
export BRAINIO_HOME="$SCRATCH/brainio"
export RESULTCACHING_HOME=$SCRATCH
cd factorized-neural-predictivity
python model_evaluation.py 0 1 2 3 4 5 6 7 8 9 10 11 12
