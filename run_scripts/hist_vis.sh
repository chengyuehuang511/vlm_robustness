#!/bin/bash

#SBATCH --job-name=hist_vis
#SBATCH --output=hist_vis.out
#SBATCH --partition="overcap"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node="a40:1"
#SBATCH --qos="short"
#SBATCH -x shakey,nestor,voltron,chappie,puma,randotron,cheetah,baymax,tachikoma,uniblab,major,optimistprime,hk47,xaea-12,dave,crushinator,kitt




export PYTHONUNBUFFERED=TRUE
source /nethome/bmaneech3/flash/miniconda3/bin/activate riplenv

conda activate riplenv
cd /nethome/bmaneech3/flash/vlm_robustness

srun -u python -m torch.distributed.run --nproc_per_node=1 /nethome/bmaneech3/flash/vlm_robustness/ood_test/contextual_ood/process_metric/histogram_vis.py --ft_method "fft" 

