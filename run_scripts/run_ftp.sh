#!/bin/bash

#SBATCH --job-name=runftp
#SBATCH --output=runftp.out
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


srun -u python -m torch.distributed.run --nproc_per_node=8 /nethome/bmaneech3/flash/vlm_robustness/ood_test/contextual_ood/test_model.py --cfg-path /nethome/bmaneech3/flash/vlm_robustness/configs/ftp/advqa_test.yaml

srun -u python -m torch.distributed.run --nproc_per_node=8 /nethome/bmaneech3/flash/vlm_robustness/ood_test/contextual_ood/test_model.py --cfg-path /nethome/bmaneech3/flash/vlm_robustness/configs/ftp/cvvqa_test.yaml

srun -u python -m torch.distributed.run --nproc_per_node=8 /nethome/bmaneech3/flash/vlm_robustness/ood_test/contextual_ood/test_model.py --cfg-path /nethome/bmaneech3/flash/vlm_robustness/configs/ftp/ivvqa_test.yaml

srun -u python -m torch.distributed.run --nproc_per_node=8 /nethome/bmaneech3/flash/vlm_robustness/ood_test/contextual_ood/test_model.py --cfg-path /nethome/bmaneech3/flash/vlm_robustness/configs/ftp/okvqa_test.yaml

srun -u python -m torch.distributed.run --nproc_per_node=8 /nethome/bmaneech3/flash/vlm_robustness/ood_test/contextual_ood/test_model.py --cfg-path /nethome/bmaneech3/flash/vlm_robustness/configs/ftp/textvqa_test.yaml

srun -u python -m torch.distributed.run --nproc_per_node=8 /nethome/bmaneech3/flash/vlm_robustness/ood_test/contextual_ood/test_model.py --cfg-path /nethome/bmaneech3/flash/vlm_robustness/configs/ftp/vqace_test.yaml

srun -u python -m torch.distributed.run --nproc_per_node=8 /nethome/bmaneech3/flash/vlm_robustness/ood_test/contextual_ood/test_model.py --cfg-path /nethome/bmaneech3/flash/vlm_robustness/configs/ftp/vqacp_test.yaml

srun -u python -m torch.distributed.run --nproc_per_node=8 /nethome/bmaneech3/flash/vlm_robustness/ood_test/contextual_ood/test_model.py --cfg-path /nethome/bmaneech3/flash/vlm_robustness/configs/ftp/vqarep_test.yaml

srun -u python -m torch.distributed.run --nproc_per_node=8 /nethome/bmaneech3/flash/vlm_robustness/ood_test/contextual_ood/test_model.py --cfg-path /nethome/bmaneech3/flash/vlm_robustness/configs/ftp/vqav2_val.yaml

srun -u python -m torch.distributed.run --nproc_per_node=8 /nethome/bmaneech3/flash/vlm_robustness/ood_test/contextual_ood/test_model.py --cfg-path /nethome/bmaneech3/flash/vlm_robustness/configs/ftp/vqav2_train.yaml

srun -u python -m torch.distributed.run --nproc_per_node=8 /nethome/bmaneech3/flash/vlm_robustness/ood_test/contextual_ood/test_model.py --cfg-path /nethome/bmaneech3/flash/vlm_robustness/configs/ftp/vizwiz_test.yaml


