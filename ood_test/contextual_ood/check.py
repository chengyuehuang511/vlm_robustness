import torch
import os

# Specify the path to your .pth file
# file_path = '/coc/pskynet4/bmaneech3/vlm_robustness/result_output/contextual_ood/pt_indiv_result/advqa_test_image.pth'

# folder = '/coc/pskynet4/bmaneech3/vlm_robustness/result_output/contextual_ood/ft_indiv_result/fft/'
folder = '/coc/pskynet4/bmaneech3/vlm_robustness/result_output/contextual_ood/pt_indiv_result/'
# go over all the files in the folder
for file in os.listdir(folder):
    print(file)
    file_path = os.path.join(folder, file)
    # Load the .pth file
    data = torch.load(file_path)
    # print(len(data['image']))
    print(len(data))