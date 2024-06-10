#!/bin/bash
cd /nethome/chuang475/flash/projects/vlm_robustness
name="test_albef_vqacp"

job_name="${name}_$(date +%Y%m%d_%H%M%S)"
output_dir="output/${job_name}"
mkdir -p "$output_dir"
sbatch --export "ALL" --job-name="${job_name}" --output="${output_dir}/slurm-%j.out" --error="${output_dir}/slurm-%j.err" ${name}.sh
