#!/bin/bash

#SBATCH --job-name=runlpft_2
#SBATCH --output=runlpft_2.out
#SBATCH --partition="overcap"
#SBATCH --nodes=1
#SBATCH --gpus-per-node="a40:8"
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=TRUE
source /nethome/bmaneech3/flash/miniconda3/bin/activate riplenv

conda activate riplenv
cd /nethome/bmaneech3/flash/vlm_robustness


srun -u python -m torch.distributed.run --nproc_per_node=8 /nethome/bmaneech3/flash/vlm_robustness/ood_test/contextual_ood/test_model.py --cfg-path /nethome/bmaneech3/flash/vlm_robustness/configs/lpft/advqa_test.yaml

srun -u python -m torch.distributed.run --nproc_per_node=8 /nethome/bmaneech3/flash/vlm_robustness/ood_test/contextual_ood/test_model.py --cfg-path /nethome/bmaneech3/flash/vlm_robustness/configs/lpft/cvvqa_test.yaml

srun -u python -m torch.distributed.run --nproc_per_node=8 /nethome/bmaneech3/flash/vlm_robustness/ood_test/contextual_ood/test_model.py --cfg-path /nethome/bmaneech3/flash/vlm_robustness/configs/lpft/ivvqa_test.yaml

srun -u python -m torch.distributed.run --nproc_per_node=8 /nethome/bmaneech3/flash/vlm_robustness/ood_test/contextual_ood/test_model.py --cfg-path /nethome/bmaneech3/flash/vlm_robustness/configs/lpft/okvqa_test.yaml

srun -u python -m torch.distributed.run --nproc_per_node=8 /nethome/bmaneech3/flash/vlm_robustness/ood_test/contextual_ood/test_model.py --cfg-path /nethome/bmaneech3/flash/vlm_robustness/configs/lpft/textvqa_test.yaml

srun -u python -m torch.distributed.run --nproc_per_node=8 /nethome/bmaneech3/flash/vlm_robustness/ood_test/contextual_ood/test_model.py --cfg-path /nethome/bmaneech3/flash/vlm_robustness/configs/lpft/vqace_test.yaml

srun -u python -m torch.distributed.run --nproc_per_node=8 /nethome/bmaneech3/flash/vlm_robustness/ood_test/contextual_ood/test_model.py --cfg-path /nethome/bmaneech3/flash/vlm_robustness/configs/lpft/vqacp_test.yaml

srun -u python -m torch.distributed.run --nproc_per_node=8 /nethome/bmaneech3/flash/vlm_robustness/ood_test/contextual_ood/test_model.py --cfg-path /nethome/bmaneech3/flash/vlm_robustness/configs/lpft/vqarep_test.yaml

srun -u python -m torch.distributed.run --nproc_per_node=8 /nethome/bmaneech3/flash/vlm_robustness/ood_test/contextual_ood/test_model.py --cfg-path /nethome/bmaneech3/flash/vlm_robustness/configs/lpft/vqav2_val.yaml

srun -u python -m torch.distributed.run --nproc_per_node=8 /nethome/bmaneech3/flash/vlm_robustness/ood_test/contextual_ood/test_model.py --cfg-path /nethome/bmaneech3/flash/vlm_robustness/configs/lpft/vqav2_train.yaml

srun -u python -m torch.distributed.run --nproc_per_node=8 /nethome/bmaneech3/flash/vlm_robustness/ood_test/contextual_ood/test_model.py --cfg-path /nethome/bmaneech3/flash/vlm_robustness/configs/lpft/vizwiz_test.yaml


