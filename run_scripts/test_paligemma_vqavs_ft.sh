#!/bin/bash
#SBATCH --job-name=vqavs_ft_test_paligemma
#SBATCH --output=vqavs_ft_test_paligemma.out
#SBATCH --error=vqavs_ft_test_paligemma.err
#SBATCH --partition="overcap"
#SBATCH --nodes=1
#SBATCH --gpus-per-node="a40:8"
#SBATCH --qos="short"
#SBATCH -x shakey,nestor,voltron,chappie,puma,randotron,cheetah,baymax,tachikoma,uniblab,major,optimistprime,hk47,xaea-12,dave,crushinator,kitt,gundam
#SBATCH --mem-per-gpu=45G
source /nethome/bmaneech3/flash/miniconda3/bin/activate riplenv

conda activate riplenv
cd /nethome/bmaneech3/flash/vlm_robustness


srun -u python -m torch.distributed.run --nproc_per_node=8 evaluate.py --cfg-path /nethome/bmaneech3/flash/vlm_robustness/configs/paligemma/vqavs_ft_test.yaml





