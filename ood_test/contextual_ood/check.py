import torch
import os
import pandas as pd

# Specify the path to your .pth file
# file_path = '/coc/pskynet4/bmaneech3/vlm_robustness/result_output/contextual_ood/pt_indiv_result/advqa_test_image.pth'

# folder = '/coc/pskynet4/bmaneech3/vlm_robustness/result_output/contextual_ood/ft_indiv_result/fft/'
# folder = '/coc/pskynet4/bmaneech3/vlm_robustness/result_output/contextual_ood/pt_indiv_result/'
folder = '/coc/testnvme/chuang475/projects/vlm_robustness/ood_test/contextual_ood/xttn_results/fft'
intersect = []
i = 0
# go over all the files in the folder
for file in os.listdir(folder):
    print(file)
    coco_datasets = ['coco_vqa_ce_val.csv', 'coco_vqa_rephrasings_val.csv', 'coco_vqa_raw_val.csv', 'coco_okvqa_val.csv', 'coco_iv-vqa_val.csv', 'coco_cv-vqa_val.csv', 'coco_advqa_val.csv', 'coco_vqa_cp_val.csv']  # 'coco_advqa_val.csv', 'coco_vqa_cp_val.csv', 
    if file in coco_datasets:
        file_path = os.path.join(folder, file)
        # # Load the .pth file
        # data = torch.load(file_path)
        # Load the .csv file
        data = pd.read_csv(file_path)
        image_path = data['image_path']
        image_path = image_path.str.split('/').str[-1]
        # print(len(data['image']))
        if file == 'coco_cv-vqa_val.csv' or file == 'coco_iv-vqa_val.csv':
            # e.g. change /coc/pskynet4/chuang475/projects/vlm_robustness/tmp/datasets/cv-vqa/val/BS/vedika2/nobackup/thesis/IMAGES_counting_del1_edited_VQA_v2/val2014/COCO_val2014_000000262162_000000376154.jpg to /coc/pskynet6/chuang475/.cache/lavis/coco/images/val2014/COCO_val2014_000000262162.jpg
            image_path = image_path.str.split('/').str[-1].str.split('_').str[:-1].str.join('_') + '.jpg'
        # print(image_path[0])
        image_path = image_path.str.split('_').str[-1]
        if i == 0:
            intersect = image_path
        else:
            intersect = intersect[intersect.isin(image_path)]
        i += 1
intersect = set(intersect)
for i in intersect:
    print(f"/coc/pskynet6/chuang475/.cache/lavis/coco/images/val2014/COCO_val2014_{i}")

"""
/coc/pskynet6/chuang475/.cache/lavis/coco/images/val2014/COCO_val2014_000000531707.jpg
/coc/pskynet6/chuang475/.cache/lavis/coco/images/val2014/COCO_val2014_000000419408.jpg
/coc/pskynet6/chuang475/.cache/lavis/coco/images/val2014/COCO_val2014_000000507575.jpg
/coc/pskynet6/chuang475/.cache/lavis/coco/images/val2014/COCO_val2014_000000383339.jpg
/coc/pskynet6/chuang475/.cache/lavis/coco/images/val2014/COCO_val2014_000000112110.jpg
/coc/pskynet6/chuang475/.cache/lavis/coco/images/val2014/COCO_val2014_000000281759.jpg
/coc/pskynet6/chuang475/.cache/lavis/coco/images/val2014/COCO_val2014_000000409630.jpg
/coc/pskynet6/chuang475/.cache/lavis/coco/images/val2014/COCO_val2014_000000029984.jpg
/coc/pskynet6/chuang475/.cache/lavis/coco/images/val2014/COCO_val2014_000000235784.jpg
/coc/pskynet6/chuang475/.cache/lavis/coco/images/val2014/COCO_val2014_000000428454.jpg
/coc/pskynet6/chuang475/.cache/lavis/coco/images/val2014/COCO_val2014_000000119088.jpg
/coc/pskynet6/chuang475/.cache/lavis/coco/images/val2014/COCO_val2014_000000493772.jpg
/coc/pskynet6/chuang475/.cache/lavis/coco/images/val2014/COCO_val2014_000000522713.jpg
/coc/pskynet6/chuang475/.cache/lavis/coco/images/val2014/COCO_val2014_000000214753.jpg
/coc/pskynet6/chuang475/.cache/lavis/coco/images/val2014/COCO_val2014_000000111609.jpg
/coc/pskynet6/chuang475/.cache/lavis/coco/images/val2014/COCO_val2014_000000222317.jpg
/coc/pskynet6/chuang475/.cache/lavis/coco/images/val2014/COCO_val2014_000000209613.jpg
/coc/pskynet6/chuang475/.cache/lavis/coco/images/val2014/COCO_val2014_000000443844.jpg
/coc/pskynet6/chuang475/.cache/lavis/coco/images/val2014/COCO_val2014_000000415194.jpg
/coc/pskynet6/chuang475/.cache/lavis/coco/images/val2014/COCO_val2014_000000007281.jpg
/coc/pskynet6/chuang475/.cache/lavis/coco/images/val2014/COCO_val2014_000000017714.jpg
/coc/pskynet6/chuang475/.cache/lavis/coco/images/val2014/COCO_val2014_000000184321.jpg
/coc/pskynet6/chuang475/.cache/lavis/coco/images/val2014/COCO_val2014_000000172935.jpg
/coc/pskynet6/chuang475/.cache/lavis/coco/images/val2014/COCO_val2014_000000149568.jpg
"""