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
!#SBATCH --signal=SIGUSR1@90

module load conda
conda activate ml-tt

#export NCCL_DEBUG=INFO
#export PYTHONFAULTHANDLER=1

# srun --label --cpu-bind=v --accel-bind=v python -u training.py
srun python training.py
