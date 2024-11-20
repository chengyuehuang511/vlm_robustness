#!/bin/bash
cd /coc/testnvme/chuang475/projects/vlm_robustness/
name="run_fft"
method="fft"

for method in "lpft" "pt_emb" "fft"  #"spd" "ftp" "lora" "lp" 
do
    job_name="${name}_$(date +%Y%m%d_%H%M%S)"
    output_dir="tpcgrad_share6/output/vqa/attention_results/${job_name}"
    mkdir -p "$output_dir"
    sbatch --export "ALL,method=$method" --job-name="${job_name}" --output="${output_dir}/slurm-%j.out" --error="${output_dir}/slurm-%j.err" run_scripts/xattn/${name}.sh
    sleep 5
done