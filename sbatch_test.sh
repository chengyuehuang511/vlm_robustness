#!/bin/bash
cd /nethome/chuang475/flash/projects/vlm_robustness

for name in "val_paligemma_vqa" #"test_paligemma_vqa_ce" "test_paligemma_vqacp" "test_paligemma_vqa_rep" "test_paligemma_ok-vqa" "test_paligemma_vizwiz" "test_paligemma_textvqa" "test_paligemma_advqa" "test_paligemma_cv-vqa" "test_paligemma_iv-vqa"
do
    job_name="${name}_$(date +%Y%m%d_%H%M%S)"
    output_dir="output/evaluate_wise/${job_name}"
    mkdir -p "$output_dir"
    sbatch --export "ALL" --job-name="${job_name}" --output="${output_dir}/slurm-%j.out" --error="${output_dir}/slurm-%j.err" ${name}.sh
done

