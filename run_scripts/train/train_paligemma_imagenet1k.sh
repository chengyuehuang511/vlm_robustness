#!/bin/bash

#SBATCH --partition="kira-lab"
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=6
#SBATCH --gpus-per-node="a40:8"
#SBATCH --qos="long"
#SBATCH -x shakey,nestor,voltron,chappie,puma,randotron,cheetah,baymax,tachikoma,uniblab,optimistprime,hk47,ig-88,omgwth,qt-1,sonny
#SBATCH --mem-per-gpu=45G

cd /nethome/chuang475/flash/projects/vlm_robustness/

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=42
export TOKENIZERS_PARALLELISM=false

srun -u /nethome/chuang475/flash/miniconda3/envs/lavis/bin/python -m torch.distributed.run --nproc_per_node=8 train.py --cfg-path configs/paligemma/imagenet1k_train.yaml