#!/bin/bash

#SBATCH --job-name=report_metrics
#SBATCH --output=report_metrics.out
#SBATCH --partition="kira-lab"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7


export PYTHONUNBUFFERED=TRUE
source /nethome/bmaneech3/flash/miniconda3/bin/activate riplenv

conda activate riplenv
cd /nethome/bmaneech3/flash/vlm_robustness


srun -u python -u /nethome/bmaneech3/flash/vlm_robustness/ood_test/measure.py