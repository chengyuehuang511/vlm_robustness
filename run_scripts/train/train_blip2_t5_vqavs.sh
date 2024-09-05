#!/bin/bash
#SBATCH --job-name=vqa_vs_blip2t5
#SBATCH --output=vqa_vs_blip2t5.out
#SBATCH --error=vqa_vs_blip2t5.err
#SBATCH --partition="overcap"
#SBATCH --nodes=1
#SBATCH --gpus-per-node="a40:8"
#SBATCH --qos="short"
#SBATCH -x shakey,nestor,voltron,chappie,puma,randotron,cheetah,baymax,tachikoma,uniblab,major,optimistprime,hk47,xaea-12,dave,crushinator,kitt,gundam
#SBATCH --mem-per-gpu=45G
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source /nethome/bmaneech3/flash/miniconda3/bin/activate riplenv


conda activate riplenv
cd /nethome/bmaneech3/flash/vlm_robustness


srun -u python -m torch.distributed.run --nproc_per_node=8 train.py --cfg-path /nethome/bmaneech3/flash/vlm_robustness/configs/blip2/vqavs_train_t5.yaml