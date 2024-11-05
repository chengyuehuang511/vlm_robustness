#!/bin/bash

#SBATCH --job-name=run_pretrain
#SBATCH --output=run_pretrain.out
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


srun -u python -m torch.distributed.run --nproc_per_node=8 /nethome/bmaneech3/flash/vlm_robustness/ood_test/contextual_ood/test_model.py --cfg-path /nethome/bmaneech3/flash/vlm_robustness/configs/pt_emb/advqa_test.yaml

srun -u python -m torch.distributed.run --nproc_per_node=8 /nethome/bmaneech3/flash/vlm_robustness/ood_test/contextual_ood/test_model.py --cfg-path /nethome/bmaneech3/flash/vlm_robustness/configs/pt_emb/cvvqa_test.yaml

srun -u python -m torch.distributed.run --nproc_per_node=8 /nethome/bmaneech3/flash/vlm_robustness/ood_test/contextual_ood/test_model.py --cfg-path /nethome/bmaneech3/flash/vlm_robustness/configs/pt_emb/ivvqa_test.yaml

srun -u python -m torch.distributed.run --nproc_per_node=8 /nethome/bmaneech3/flash/vlm_robustness/ood_test/contextual_ood/test_model.py --cfg-path /nethome/bmaneech3/flash/vlm_robustness/configs/pt_emb/okvqa_test.yaml

srun -u python -m torch.distributed.run --nproc_per_node=8 /nethome/bmaneech3/flash/vlm_robustness/ood_test/contextual_ood/test_model.py --cfg-path /nethome/bmaneech3/flash/vlm_robustness/configs/pt_emb/textvqa_test.yaml

srun -u python -m torch.distributed.run --nproc_per_node=8 /nethome/bmaneech3/flash/vlm_robustness/ood_test/contextual_ood/test_model.py --cfg-path /nethome/bmaneech3/flash/vlm_robustness/configs/pt_emb/vqace_test.yaml

srun -u python -m torch.distributed.run --nproc_per_node=8 /nethome/bmaneech3/flash/vlm_robustness/ood_test/contextual_ood/test_model.py --cfg-path /nethome/bmaneech3/flash/vlm_robustness/configs/pt_emb/vqacp_test.yaml

srun -u python -m torch.distributed.run --nproc_per_node=8 /nethome/bmaneech3/flash/vlm_robustness/ood_test/contextual_ood/test_model.py --cfg-path /nethome/bmaneech3/flash/vlm_robustness/configs/pt_emb/vqarep_test.yaml

srun -u python -m torch.distributed.run --nproc_per_node=8 /nethome/bmaneech3/flash/vlm_robustness/ood_test/contextual_ood/test_model.py --cfg-path /nethome/bmaneech3/flash/vlm_robustness/configs/pt_emb/vqav2_val.yaml

srun -u python -m torch.distributed.run --nproc_per_node=8 /nethome/bmaneech3/flash/vlm_robustness/ood_test/contextual_ood/test_model.py --cfg-path /nethome/bmaneech3/flash/vlm_robustness/configs/pt_emb/vqav2_train.yaml

srun -u python -m torch.distributed.run --nproc_per_node=8 /nethome/bmaneech3/flash/vlm_robustness/ood_test/contextual_ood/test_model.py --cfg-path /nethome/bmaneech3/flash/vlm_robustness/configs/pt_emb/vizwiz_test.yaml



