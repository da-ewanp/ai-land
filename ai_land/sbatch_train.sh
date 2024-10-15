#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=220Gb
#SBATCH --time=48:00:00
#SBATCH --exclude=ac6-303
#SBATCH --output=slurm/test-pyt.%j.out
#SBATCH --error=slurm/test-pyt.%j.out
!#SBATCH --signal=SIGUSR1@90

module load conda
conda activate ml-tt
# conda activate ai-land

echo $CUDA_VISIBLE_DEVICES
nvidia-smi

srun python training.py
