#!/bin/bash

#SBATCH --partition="overcap"
#SBATCH --nodes=1
#SBATCH --gpus-per-node="a40:8"
#SBATCH --qos="short"
#SBATCH -x shakey,nestor,voltron,chappie,puma,randotron,cheetah,baymax,tachikoma,uniblab,major,optimistprime,hk47,xaea-12,dave,crushinator,kitt,synapse,dendrite
#SBATCH --mem-per-gpu=45G

cd /coc/testnvme/chuang475/projects/vlm_robustness/
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=42
export TOKENIZERS_PARALLELISM=false
srun -u /coc/testnvme/chuang475/miniconda3/envs/lavis_same/bin/python -m torch.distributed.run --nproc_per_node=8 evaluate.py --cfg-path configs/paligemma/cv-vqa_test.yaml