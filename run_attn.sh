#!/bin/bash
cd /coc/testnvme/chuang475/projects/vlm_robustness/
name="attention_score"

job_name="${name}_$(date +%Y%m%d_%H%M%S)"
output_dir="tpcgrad_share6/output/vqa/attention_results/${job_name}"
mkdir -p "$output_dir"
sbatch --export "ALL" --job-name="${job_name}" --output="${output_dir}/slurm-%j.out" --error="${output_dir}/slurm-%j.err" run_scripts/${name}.sh