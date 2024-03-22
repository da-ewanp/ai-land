#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=220Gb
#SBATCH --time=48:00:00
#SBATCH --exclude=ac6-303
#SBATCH --output=slurm/test-pyt.%j.out
#SBATCH --error=slurm/test-pyt.%j.out

module load conda
conda activate pl-nn

srun python training.py