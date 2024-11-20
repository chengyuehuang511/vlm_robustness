#!/bin/bash

#SBATCH --partition="overcap"
#SBATCH --nodes=1
#SBATCH --gpus-per-node="a40:2"
#SBATCH --qos="short"
#SBATCH -x shakey,nestor,voltron,chappie,puma,randotron,cheetah,baymax,tachikoma,uniblab,major,optimistprime,hk47,xaea-12,dave,crushinator,trublu
#SBATCH --mem-per-gpu=45G
#SBATCH --cpus-per-gpu=16

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=TRUE
export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=0
export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

# export MASTER_PORT=$(shuf -i 0-65535 -n 1)
# Loop until we find an unused port
while :
do
    MASTER_PORT=$(shuf -i 1024-65535 -n 1)  # Randomly choose a port between 1024 and 65535
    # Check if the port is in use
    if ! lsof -i :$MASTER_PORT > /dev/null; then
        export MASTER_PORT
        echo "Selected available port: $MASTER_PORT"
        break
    else
        echo "Port $MASTER_PORT is already in use. Trying another..."
    fi
done

export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`

echo go $COUNT_NODE
echo $HOSTNAMES

cd /coc/testnvme/chuang475/projects/vlm_robustness


srun -u /coc/testnvme/chuang475/miniconda3/envs/lavis_same/bin/python -m torch.distributed.run --nproc_per_node=2 /coc/testnvme/chuang475/projects/vlm_robustness/ood_test/contextual_ood/xattn_ddp.py --cfg-path /coc/pskynet4/bmaneech3/vlm_robustness/configs/$method/advqa_test.yaml

srun -u /coc/testnvme/chuang475/miniconda3/envs/lavis_same/bin/python -m torch.distributed.run --nproc_per_node=2 /coc/testnvme/chuang475/projects/vlm_robustness/ood_test/contextual_ood/xattn_ddp.py --cfg-path /coc/pskynet4/bmaneech3/vlm_robustness/configs/$method/cvvqa_test.yaml

srun -u /coc/testnvme/chuang475/miniconda3/envs/lavis_same/bin/python -m torch.distributed.run --nproc_per_node=2 /coc/testnvme/chuang475/projects/vlm_robustness/ood_test/contextual_ood/xattn_ddp.py --cfg-path /coc/pskynet4/bmaneech3/vlm_robustness/configs/$method/ivvqa_test.yaml

srun -u /coc/testnvme/chuang475/miniconda3/envs/lavis_same/bin/python -m torch.distributed.run --nproc_per_node=2 /coc/testnvme/chuang475/projects/vlm_robustness/ood_test/contextual_ood/xattn_ddp.py --cfg-path /coc/pskynet4/bmaneech3/vlm_robustness/configs/$method/okvqa_test.yaml

srun -u /coc/testnvme/chuang475/miniconda3/envs/lavis_same/bin/python -m torch.distributed.run --nproc_per_node=2 /coc/testnvme/chuang475/projects/vlm_robustness/ood_test/contextual_ood/xattn_ddp.py --cfg-path /coc/pskynet4/bmaneech3/vlm_robustness/configs/$method/textvqa_test.yaml

srun -u /coc/testnvme/chuang475/miniconda3/envs/lavis_same/bin/python -m torch.distributed.run --nproc_per_node=2 /coc/testnvme/chuang475/projects/vlm_robustness/ood_test/contextual_ood/xattn_ddp.py --cfg-path /coc/pskynet4/bmaneech3/vlm_robustness/configs/$method/vqace_test.yaml

srun -u /coc/testnvme/chuang475/miniconda3/envs/lavis_same/bin/python -m torch.distributed.run --nproc_per_node=2 /coc/testnvme/chuang475/projects/vlm_robustness/ood_test/contextual_ood/xattn_ddp.py --cfg-path /coc/pskynet4/bmaneech3/vlm_robustness/configs/$method/vqacp_test.yaml

srun -u /coc/testnvme/chuang475/miniconda3/envs/lavis_same/bin/python -m torch.distributed.run --nproc_per_node=2 /coc/testnvme/chuang475/projects/vlm_robustness/ood_test/contextual_ood/xattn_ddp.py --cfg-path /coc/pskynet4/bmaneech3/vlm_robustness/configs/$method/vqarep_test.yaml

srun -u /coc/testnvme/chuang475/miniconda3/envs/lavis_same/bin/python -m torch.distributed.run --nproc_per_node=2 /coc/testnvme/chuang475/projects/vlm_robustness/ood_test/contextual_ood/xattn_ddp.py --cfg-path /coc/pskynet4/bmaneech3/vlm_robustness/configs/$method/vqav2_val.yaml

# srun -u /coc/testnvme/chuang475/miniconda3/envs/lavis_same/bin/python -m torch.distributed.run --nproc_per_node=2 /coc/testnvme/chuang475/projects/vlm_robustness/ood_test/contextual_ood/xattn_ddp.py --cfg-path /coc/pskynet4/bmaneech3/vlm_robustness/configs/$method/vqav2_train.yaml

srun -u /coc/testnvme/chuang475/miniconda3/envs/lavis_same/bin/python -m torch.distributed.run --nproc_per_node=2 /coc/testnvme/chuang475/projects/vlm_robustness/ood_test/contextual_ood/xattn_ddp.py --cfg-path /coc/pskynet4/bmaneech3/vlm_robustness/configs/$method/vizwiz_test.yaml


