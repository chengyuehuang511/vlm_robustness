#!/bin/bash
#SBATCH -A gts-smukhopadhyay6-cognisense
#SBATCH --cpus-per-gpu=8
##SBATCH --nodes=1
#SBATCH -N1 --gres=gpu:h200:4
##SBATCH --constraint=A100-80GB
##SBATCH --mem-per-gpu=80G

#SBATCH -t 24:00:00                                    # Duration of the job (Ex: 15 mins)
#SBATCH -q inferno                               # QOS Name
#SBATCH --mail-type=BEGIN,END,FAIL              # Mail preferences
#SBATCH --mail-user=chuang475@gatech.edu        # E-mail address for notifications

module load anaconda3/2022.05.0.1
# module load cuda/11.8.0
# conda init
conda activate lavis

cd /storage/home/hcoda1/8/mzhang445/p-smukhopadhyay6-0/chengyue/vlm_robustness

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=42
export TOKENIZERS_PARALLELISM=false

srun -u python -m torch.distributed.run --nnodes=1 --nproc_per_node=4 --master_port=25678 evaluate.py --cfg-path configs/paligemma/gqa_val.yaml --options run.init_lr=$init_lr run.min_lr=$min_lr run.warmup_lr=$warmup_lr run.weight_decay=$weight_decay
