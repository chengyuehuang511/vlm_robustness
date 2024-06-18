cd /nethome/chuang475/flash/projects/vlm_robustness
pip install transformers==4.27
/nethome/chuang475/flash/miniconda3/envs/lavis/bin/python -m torch.distributed.run --nproc_per_node=1 train.py --cfg-path configs/blip2/vqav2_train_t5.yaml