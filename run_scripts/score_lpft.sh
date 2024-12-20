#!/bin/bash
#SBATCH --job-name=calclpft
#SBATCH --output=calclpft.out
#SBATCH --error=calclpft.err
#SBATCH --partition="overcap"
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=6
#SBATCH --gpus-per-node="a40:8"
#SBATCH --qos="short"
#SBATCH -x shakey,nestor,voltron,chappie,puma,randotron,cheetah,baymax,tachikoma,uniblab,major,optimistprime,hk47,xaea-12,dave,crushinator,kitt,gundam
#SBATCH --mem-per-gpu=45G
source /nethome/bmaneech3/flash/miniconda3/bin/activate riplenv


conda activate riplenv
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=42
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /nethome/bmaneech3/flash/vlm_robustness

concept_list_json='[["image", "joint"], ["ques_ft"]]'

srun -u python -m torch.distributed.run --nproc_per_node=1 /nethome/bmaneech3/flash/vlm_robustness/ood_test/contextual_ood/measure.py --ft_method "lpft"  --concept_list "$concept_list_json"