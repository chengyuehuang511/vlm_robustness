#!/bin/bash

#SBATCH --partition="overcap"
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=16
#SBATCH --gpus-per-node="a40:8"
#SBATCH --qos="short"
##SBATCH -x shakey,nestor,voltron,chappie,puma,randotron,cheetah,baymax,tachikoma,uniblab,optimistprime,hk47,ig-88,omgwth,qt-1,sonny
#SBATCH --mem-per-gpu=45G

cd /coc/testnvme/chuang475/projects/vlm_robustness/

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=42
export TOKENIZERS_PARALLELISM=false
export TORCH_DISTRIBUTED_DEBUG=INFO

srun -u /coc/testnvme/chuang475/miniconda3/envs/lavis_same/bin/python -m torch.distributed.run --nproc_per_node=8 train.py --cfg-path configs/llava/domainnet-real_train.yaml