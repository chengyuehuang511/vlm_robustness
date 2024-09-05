#!/bin/bash
#SBATCH --job-name=vqa_vs_fft
#SBATCH --output=vqa_vs_fft.out
#SBATCH --error=vqa_vs_fft.err
#SBATCH --partition="overcap"
#SBATCH --nodes=1
#SBATCH --gpus-per-node="a40:8"
#SBATCH --qos="short"
#SBATCH -x shakey,nestor,voltron,chappie,puma,randotron,cheetah,baymax,tachikoma,uniblab,major,optimistprime,hk47,xaea-12,dave,crushinator,kitt,gundam
#SBATCH --mem-per-gpu=45G
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=42
export TOKENIZERS_PARALLELISM=false
source /nethome/bmaneech3/flash/miniconda3/bin/activate riplenv


conda activate riplenv
cd /nethome/bmaneech3/flash/vlm_robustness


srun -u python -m torch.distributed.run --nproc_per_node=8 train.py --cfg-path /nethome/bmaneech3/flash/vlm_robustness/configs/paligemma/vqa_vs_fft.yaml
