#!/bin/bash
#SBATCH --qos=ng
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --mem=256Gb
#SBATCH --time=48:00:00
#SBATCH --exclude=ac6-303
#SBATCH --output=slurm/test-pyt.%j.out
#SBATCH --error=slurm/test-pyt.%j.out

module load conda
conda activate ml-tt
# conda activate ai-land

echo $CUDA_VISIBLE_DEVICES
nvidia-smi

srun python training.py
