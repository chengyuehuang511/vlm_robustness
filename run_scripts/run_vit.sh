#!/bin/bash

#SBATCH --job-name=run_vit
#SBATCH --output=run_vit.out
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


srun -u python -m torch.distributed.run --nproc_per_node=8 /nethome/bmaneech3/flash/vlm_robustness/ood_test/contextual_ood/test_model.py --cfg-path /nethome/bmaneech3/flash/vlm_robustness/configs/vit/advqa_test.yaml

srun -u python -m torch.distributed.run --nproc_per_node=8 /nethome/bmaneech3/flash/vlm_robustness/ood_test/contextual_ood/test_model.py --cfg-path /nethome/bmaneech3/flash/vlm_robustness/configs/vit/cvvqa_test.yaml

srun -u python -m torch.distributed.run --nproc_per_node=8 /nethome/bmaneech3/flash/vlm_robustness/ood_test/contextual_ood/test_model.py --cfg-path /nethome/bmaneech3/flash/vlm_robustness/configs/vit/ivvqa_test.yaml

srun -u python -m torch.distributed.run --nproc_per_node=8 /nethome/bmaneech3/flash/vlm_robustness/ood_test/contextual_ood/test_model.py --cfg-path /nethome/bmaneech3/flash/vlm_robustness/configs/vit/okvqa_test.yaml

srun -u python -m torch.distributed.run --nproc_per_node=8 /nethome/bmaneech3/flash/vlm_robustness/ood_test/contextual_ood/test_model.py --cfg-path /nethome/bmaneech3/flash/vlm_robustness/configs/vit/textvqa_test.yaml

srun -u python -m torch.distributed.run --nproc_per_node=8 /nethome/bmaneech3/flash/vlm_robustness/ood_test/contextual_ood/test_model.py --cfg-path /nethome/bmaneech3/flash/vlm_robustness/configs/vit/vqace_test.yaml

srun -u python -m torch.distributed.run --nproc_per_node=8 /nethome/bmaneech3/flash/vlm_robustness/ood_test/contextual_ood/test_model.py --cfg-path /nethome/bmaneech3/flash/vlm_robustness/configs/vit/vqacp_test.yaml

srun -u python -m torch.distributed.run --nproc_per_node=8 /nethome/bmaneech3/flash/vlm_robustness/ood_test/contextual_ood/test_model.py --cfg-path /nethome/bmaneech3/flash/vlm_robustness/configs/vit/vqarep_test.yaml

srun -u python -m torch.distributed.run --nproc_per_node=8 /nethome/bmaneech3/flash/vlm_robustness/ood_test/contextual_ood/test_model.py --cfg-path /nethome/bmaneech3/flash/vlm_robustness/configs/vit/vqav2_val.yaml

srun -u python -m torch.distributed.run --nproc_per_node=8 /nethome/bmaneech3/flash/vlm_robustness/ood_test/contextual_ood/test_model.py --cfg-path /nethome/bmaneech3/flash/vlm_robustness/configs/vit/vqav2_train.yaml

srun -u python -m torch.distributed.run --nproc_per_node=8 /nethome/bmaneech3/flash/vlm_robustness/ood_test/contextual_ood/test_model.py --cfg-path /nethome/bmaneech3/flash/vlm_robustness/configs/vit/vizwiz_test.yaml



