#!/bin/bash                   
#SBATCH --partition="kira-lab"
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=16
#SBATCH --gpus-per-node="a40:4"
#SBATCH --qos="short"
#SBATCH -x nestor,uniblab,chappie,conroy,deebot
#SBATCH --mem-per-gpu=45G

# Automatically determine MASTER_ADDR and MASTER_PORT
export MASTER_ADDR=$(hostname)  # Use the hostname of the current node

# Assign a unique port for each job in the array
PORT_BASE=29500                 # Base port number
JOB_OFFSET=$SLURM_JOB_ID # Unique offset for the job array
PORT=$((SLURM_JOB_ID % 65536))
export MASTER_PORT=$PORT

# Debugging info
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"

cd /coc/testnvme/chuang475/projects/vlm_robustness/
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=42
export TOKENIZERS_PARALLELISM=false
srun -u /coc/testnvme/chuang475/miniconda3/envs/lavis_same/bin/python -m torch.distributed.run --nproc_per_node=4 evaluate.py --cfg-path configs/llava/domainnet-real_test.yaml
