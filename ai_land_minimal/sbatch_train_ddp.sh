#!/bin/bash
#SBATCH --qos=ng
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=228GB
#SBATCH --time=48:00:00
#SBATCH --account=ecaifs
#SBATCH --output=slurm/test-ddp.%j.out
#SBATCH --error=slurm/test-ddp.%j.out

module load conda
conda activate ml-tt
# conda activate ai-land

export NCCL_DEBUG=INFO
#export PYTHONFAULTHANDLER=1
#export CUDA_VISIBLE_DEVICES=4

# srun --label --cpu-bind=v --accel-bind=v python -u training.py
# srun python training.py
python training.py
