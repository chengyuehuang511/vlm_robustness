#!/bin/bash

#SBATCH --partition="kira-lab"
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=16
#SBATCH --gpus-per-node="a40:8"
#SBATCH --qos="short"
#SBATCH -x nestor,crushinator,robby
#SBATCH --mem-per-gpu=45G

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

cd /coc/testnvme/chuang475/projects/vlm_robustness/

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=42
export TOKENIZERS_PARALLELISM=false
export TORCH_DISTRIBUTED_DEBUG=INFO

# /nethome/chuang475/flash/miniconda3/envs/lavis
srun -u /coc/testnvme/chuang475/miniconda3/envs/lavis_same/bin/python -m torch.distributed.run --nproc_per_node=8 train.py --cfg-path configs/llava/vqav2_train.yaml