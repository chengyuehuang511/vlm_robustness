#!/bin/bash
cd /coc/testnvme/chuang475/projects/vlm_robustness/
# name="train_paligemma_vqa"
# name="train_paligemma_domainnet-real"
# name="train_paligemma_imagenet1k"
name="train_llava_domainnet-real"

job_name="${name}_$(date +%Y%m%d_%H%M%S)"
output_dir="tpcgrad_share6/output/llava/domainnet/ft/${job_name}"
mkdir -p "$output_dir"
sbatch --export "ALL" --job-name="${job_name}" --output="${output_dir}/slurm-%j.out" --error="${output_dir}/slurm-%j.err" run_scripts/train/${name}.sh