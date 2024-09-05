#!/bin/bash
cd /nethome/chuang475/flash/projects/vlm_robustness
# name="train_paligemma_vqa"
name="train_paligemma_domainnet-real"
# name="train_paligemma_imagenet1k"

job_name="${name}_$(date +%Y%m%d_%H%M%S)"
output_dir="output/domainnet/pcgrad_1e-3_1e-4_1/${job_name}"
mkdir -p "$output_dir"
sbatch --export "ALL" --job-name="${job_name}" --output="${output_dir}/slurm-%j.out" --error="${output_dir}/slurm-%j.err" run_scripts/train/${name}.sh