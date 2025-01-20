#!/bin/bash
cd /coc/testnvme/chuang475/projects/vlm_robustness/

for wise in 0.1 0.2 0.3 0.9
do
    for name in "val_paligemma_vqa" "test_paligemma_vqa_ce" "test_paligemma_vqacp" "test_paligemma_vqa_rep" "test_paligemma_ok-vqa" "test_paligemma_vizwiz" "test_paligemma_textvqa" "test_paligemma_advqa" "test_paligemma_cv-vqa" "test_paligemma_iv-vqa"
    # "test_llava_vqa_ce" "test_llava_vqacp" "test_llava_vqa_rep" "test_llava_ok-vqa" "test_llava_vizwiz" "test_llava_textvqa" "test_llava_advqa" "test_llava_cv-vqa" "test_llava_iv-vqa" 
    #  "domainnet-infograph_llava_test" "domainnet-painting_llava_test" "domainnet-real_llava_test" "domainnet-clipart_llava_test" "domainnet-sketch_llava_test" "domainnet-quickdraw_llava_test"
    # "domainnet-clipart_paligemma_test" "domainnet-infograph_paligemma_test" "domainnet-painting_paligemma_test" "domainnet-quickdraw_paligemma_test" "domainnet-real_paligemma_test" "domainnet-sketch_paligemma_test"
    #"test_florence2_imagenet-2" "test_florence2_imagenet1k" "test_florence2_imagenet-r" "test_florence2_imagenet-a" "test_florence2_imagenet-s" 
    #"val_florence2_vqa" "test_florence2_vqa_ce" "test_florence2_vqacp" "test_florence2_vqa_rep" "test_florence2_ok-vqa" "test_florence2_vizwiz" "test_florence2_textvqa" "test_florence2_advqa" "test_florence2_cv-vqa" "test_florence2_iv-vqa"
    #"test_paligemma_imagenet1k" "test_paligemma_imagenet-2" "test_paligemma_imagenet-r" "test_paligemma_imagenet-a" "test_paligemma_imagenet-s" 
    #"val_paligemma_vqa" "test_paligemma_vqa_ce" "test_paligemma_vqacp" "test_paligemma_vqa_rep" "test_paligemma_ok-vqa" "test_paligemma_vizwiz" "test_paligemma_textvqa" "test_paligemma_advqa" "test_paligemma_cv-vqa" "test_paligemma_iv-vqa"
    do
        job_name="${name}_$(date +%Y%m%d_%H%M%S)"
        output_dir="tpcgrad_share6/output/paligemma/vqa_ft/wise_lora/${wise}/${job_name}"
        mkdir -p "$output_dir"
        sbatch --export "ALL,wise=${wise}" --job-name="${job_name}" --output="${output_dir}/slurm-%j.out" --error="${output_dir}/slurm-%j.err" run_scripts/test/paligemma/${name}.sh
    done
done

