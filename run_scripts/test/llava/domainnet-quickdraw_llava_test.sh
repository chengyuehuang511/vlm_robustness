#!/bin/bash                   
#SBATCH --partition="kira-lab"
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=6
#SBATCH --gpus-per-node="a40:2"
#SBATCH --qos="short"
#SBATCH -x shakey,nestor,voltron,chappie,puma,randotron,cheetah,baymax,tachikoma,uniblab,major,optimistprime,hk47,xaea-12,dave,crushinator,kitt
#SBATCH --mem-per-gpu=45G

cd /coc/testnvme/chuang475/projects/vlm_robustness/
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=42
export TOKENIZERS_PARALLELISM=false
srun -u /coc/testnvme/chuang475/miniconda3/envs/lavis_same/bin/python -m torch.distributed.run --nproc_per_node=2 evaluate.py --cfg-path configs/llava/domainnet-quickdraw_test.yaml
