#!/bin/bash

#SBATCH --job-name=run_spd
#SBATCH --output=run_spd.out
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


srun -u python -m torch.distributed.run --nproc_per_node=8 /nethome/bmaneech3/flash/vlm_robustness/ood_test/contextual_ood/test_model.py --cfg-path /nethome/bmaneech3/flash/vlm_robustness/configs/spd/advqa_test.yaml

srun -u python -m torch.distributed.run --nproc_per_node=8 /nethome/bmaneech3/flash/vlm_robustness/ood_test/contextual_ood/test_model.py --cfg-path /nethome/bmaneech3/flash/vlm_robustness/configs/spd/cvvqa_test.yaml

srun -u python -m torch.distributed.run --nproc_per_node=8 /nethome/bmaneech3/flash/vlm_robustness/ood_test/contextual_ood/test_model.py --cfg-path /nethome/bmaneech3/flash/vlm_robustness/configs/spd/ivvqa_test.yaml

srun -u python -m torch.distributed.run --nproc_per_node=8 /nethome/bmaneech3/flash/vlm_robustness/ood_test/contextual_ood/test_model.py --cfg-path /nethome/bmaneech3/flash/vlm_robustness/configs/spd/okvqa_test.yaml

srun -u python -m torch.distributed.run --nproc_per_node=8 /nethome/bmaneech3/flash/vlm_robustness/ood_test/contextual_ood/test_model.py --cfg-path /nethome/bmaneech3/flash/vlm_robustness/configs/spd/textvqa_test.yaml

srun -u python -m torch.distributed.run --nproc_per_node=8 /nethome/bmaneech3/flash/vlm_robustness/ood_test/contextual_ood/test_model.py --cfg-path /nethome/bmaneech3/flash/vlm_robustness/configs/spd/vqace_test.yaml

srun -u python -m torch.distributed.run --nproc_per_node=8 /nethome/bmaneech3/flash/vlm_robustness/ood_test/contextual_ood/test_model.py --cfg-path /nethome/bmaneech3/flash/vlm_robustness/configs/spd/vqacp_test.yaml

srun -u python -m torch.distributed.run --nproc_per_node=8 /nethome/bmaneech3/flash/vlm_robustness/ood_test/contextual_ood/test_model.py --cfg-path /nethome/bmaneech3/flash/vlm_robustness/configs/spd/vqarep_test.yaml

srun -u python -m torch.distributed.run --nproc_per_node=8 /nethome/bmaneech3/flash/vlm_robustness/ood_test/contextual_ood/test_model.py --cfg-path /nethome/bmaneech3/flash/vlm_robustness/configs/spd/vqav2_val.yaml

srun -u python -m torch.distributed.run --nproc_per_node=8 /nethome/bmaneech3/flash/vlm_robustness/ood_test/contextual_ood/test_model.py --cfg-path /nethome/bmaneech3/flash/vlm_robustness/configs/spd/vqav2_train.yaml

srun -u python -m torch.distributed.run --nproc_per_node=8 /nethome/bmaneech3/flash/vlm_robustness/ood_test/contextual_ood/test_model.py --cfg-path /nethome/bmaneech3/flash/vlm_robustness/configs/spd/vizwiz_test.yaml



