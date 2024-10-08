#!/bin/bash

#SBATCH --partition="overcap"
#SBATCH --cpus-per-gpu=6
#SBATCH --nodes=1
#SBATCH --gpus-per-node="a40:8"
#SBATCH --qos="short"
#SBATCH -x shakey,nestor,voltron,chappie,puma,randotron,cheetah,baymax,tachikoma,uniblab,major,optimistprime,hk47,xaea-12,dave,crushinator,kitt
#SBATCH --mem-per-gpu=45G

# print cpus-per-gpu
echo "cpus-per-gpu: $SLURM_CPUS_PER_GPU"
cd /nethome/chuang475/flash/projects/vlm_robustness/
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=42
export TOKENIZERS_PARALLELISM=false
srun -u /nethome/chuang475/flash/miniconda3/envs/lavis/bin/python -m torch.distributed.run --nproc_per_node=8 evaluate.py --cfg-path configs/florence2/vqa_ce_test.yaml