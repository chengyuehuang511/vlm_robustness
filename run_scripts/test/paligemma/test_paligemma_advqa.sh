#!/bin/bash

#SBATCH --partition="overcap"
#SBATCH --nodes=1
#SBATCH --gpus-per-node="a40:6"
#SBATCH --qos="short"
#SBATCH -x shakey,nestor,voltron,chappie,puma,randotron,cheetah,baymax,tachikoma,uniblab,major,optimistprime,hk47,xaea-12,dave,crushinator,kitt
#SBATCH --mem-per-gpu=45G

cd /nethome/chuang475/flash/projects/vlm_robustness/
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=42
export TOKENIZERS_PARALLELISM=false
srun -u /nethome/chuang475/flash/miniconda3/envs/lavis/bin/python -m torch.distributed.run --nproc_per_node=6 evaluate.py --cfg-path configs/paligemma/advqa_test.yaml