#!/bin/bash

#SBATCH --partition="kira-lab"
#SBATCH --nodes=1
#SBATCH --gpus-per-node="a40:8"
#SBATCH --qos="short"
#SBATCH -x shakey,nestor,voltron,chappie,puma,randotron,cheetah,baymax,tachikoma,uniblab,major,optimistprime,hk47,xaea-12,dave,crushinator,kitt,gundam
#SBATCH --mem-per-gpu=45G

cd /nethome/chuang475/flash/projects/vlm_robustness/
srun -u /nethome/chuang475/flash/miniconda3/envs/lavis/bin/python -m torch.distributed.run --nproc_per_node=8 train.py --cfg-path configs/florence2/vqav2_train.yaml