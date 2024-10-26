#!/bin/bash

#SBATCH --job-name=inspectdrop
#SBATCH --output=inspectdrop.out
#SBATCH --partition="overcap"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7


export PYTHONUNBUFFERED=TRUE
source /nethome/bmaneech3/flash/miniconda3/bin/activate riplenv

conda activate riplenv
cd /nethome/bmaneech3/flash/vlm_robustness


srun -u python -m torch.distributed.run --nproc_per_node=1 /nethome/bmaneech3/flash/vlm_robustness/ood_test/inspect_grad.py
