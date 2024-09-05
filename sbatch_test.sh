#!/bin/bash
cd /nethome/chuang475/flash/projects/vlm_robustness

for name in "domainnet-clipart_paligemma_test" "domainnet-infograph_paligemma_test" "domainnet-painting_paligemma_test" "domainnet-quickdraw_paligemma_test" "domainnet-real_paligemma_test" "domainnet-sketch_paligemma_test" #"val_paligemma_vqa" "test_paligemma_vqa_ce" "test_paligemma_vqacp" "test_paligemma_vqa_rep" "test_paligemma_ok-vqa" "test_paligemma_vizwiz" "test_paligemma_textvqa" "test_paligemma_advqa" "test_paligemma_cv-vqa" "test_paligemma_iv-vqa"
# "domainnet-clipart_paligemma_test" "domainnet-infograph_paligemma_test" "domainnet-painting_paligemma_test" "domainnet-quickdraw_paligemma_test" "domainnet-real_paligemma_test" "domainnet-sketch_paligemma_test"
#"test_florence2_imagenet-2" "test_florence2_imagenet1k" "test_florence2_imagenet-r" "test_florence2_imagenet-a" "test_florence2_imagenet-s" 
#"val_florence2_vqa" "test_florence2_vqa_ce" "test_florence2_vqacp" "test_florence2_vqa_rep" "test_florence2_ok-vqa" "test_florence2_vizwiz" "test_florence2_textvqa" "test_florence2_advqa" "test_florence2_cv-vqa" "test_florence2_iv-vqa"
#"test_paligemma_imagenet1k" "test_paligemma_imagenet-2" "test_paligemma_imagenet-r" "test_paligemma_imagenet-a" "test_paligemma_imagenet-s" 
#"val_paligemma_vqa" "test_paligemma_vqa_ce" "test_paligemma_vqacp" "test_paligemma_vqa_rep" "test_paligemma_ok-vqa" "test_paligemma_vizwiz" "test_paligemma_textvqa" "test_paligemma_advqa" "test_paligemma_cv-vqa" "test_paligemma_iv-vqa"
do
    job_name="${name}_$(date +%Y%m%d_%H%M%S)"
    output_dir="output/domainnet_ft/spd_more_gd/${job_name}"
    mkdir -p "$output_dir"
    sbatch --export "ALL" --job-name="${job_name}" --output="${output_dir}/slurm-%j.out" --error="${output_dir}/slurm-%j.err" run_scripts/test/paligemma/${name}.sh
done

