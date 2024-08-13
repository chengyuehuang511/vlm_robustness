#!/bin/bash
cd /nethome/chuang475/flash/projects/vlm_robustness

for name in "test_paligemma_imagenet1k" "test_paligemma_imagenet-2" "test_paligemma_imagenet1k" "test_paligemma_imagenet-r" "test_paligemma_imagenet-a" "test_paligemma_imagenet-s" 
#"val_paligemma_vqa" "test_paligemma_vqa_ce" "test_paligemma_vqacp" "test_paligemma_vqa_rep" "test_paligemma_ok-vqa" "test_paligemma_vizwiz" "test_paligemma_textvqa" "test_paligemma_advqa" "test_paligemma_cv-vqa" "test_paligemma_iv-vqa"
do
    job_name="${name}_$(date +%Y%m%d_%H%M%S)"
    output_dir="output/imagenet/${job_name}"
    mkdir -p "$output_dir"
    sbatch --export "ALL" --job-name="${job_name}" --output="${output_dir}/slurm-%j.out" --error="${output_dir}/slurm-%j.err" ${name}.sh
done

