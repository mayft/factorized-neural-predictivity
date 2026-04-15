#!/bin/bash
#SBATCH --time=1:0:0
#SBATCH --gpus=h100:4
#SBATCH --mail-user=mayafthompson@gmail.com

export RESULTCACHING_HOME=$SCRATCH
python model_scores.py