/nethome/chuang475/flash/miniconda3/envs/lavis/bin/python -m torch.distributed.run --nproc_per_node=1 -m trace -l -C . evaluate.py --cfg-path configs/albef/vqa_test.yaml
