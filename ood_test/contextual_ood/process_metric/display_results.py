import json 
f = "/nethome/bmaneech3/flash/vlm_robustness/result_output/contextual_ood/maha_score_dict.json"



import matplotlib.pyplot as plt
import pandas as pd
from adjustText import adjust_text
import json 
from scipy.stats import pearsonr
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from oauth2client.service_account import ServiceAccountCredentials
import numpy as np 
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]

import gspread
import time 
from gspread import Cell 
creds = ServiceAccountCredentials.from_json_keyfile_name('/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/credential.json', scope)

client = gspread.authorize(creds)

spreadsheet = client.open_by_key('127jqguDE0jpgrqf4zH6geLH6KkCZggxousWjPQfvJE4')


SERVICE_ACCOUNT_FILE = '/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/credential.json'
SCOPES = ['https://www.googleapis.com/auth/drive.file']
credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES)
service = build('drive', 'v3', credentials=credentials)



title_dict = {
    "image" : "Vision Shift", 
    "joint" : "V+Q Joint Shift", 
    "vqa" : "V+Q+A Joint Shift", 
    "q" :"Question Mid layer Shift",
    "pt_joint" : "Pretrain Joint Shift", 
    "pt_image" : "Pretrain Image Shift"
}

# ft_methods = ["lora", "fft","lp","lpft", "ftp","spd", "uni_question", "vit"]
# "fft", "lora", "lp", "lpft", "ftp", "spd", "digrap",
ft_methods = [ "vit", "uni_question", "pt_emb"] 
# PT 
# FFT 
# LoRA 
# LP 
# LPFT 
# FTP
# SPD 
# DiGrap 




# ViT (image)
# BERT (question)
# ft_methods = ["question"]


test_splits = [
    # "coco_vqav2_train_val",
    "coco_advqa_val", 
    "coco_cv-vqa_val",
    "coco_iv-vqa_val",
    "coco_okvqa_val",
    "coco_vqa_ce_val",
    "coco_vqa_cp_val",
    "coco_vqa_raw_val",
    "coco_vqa_rephrasings_val",
    "textvqa_val",
    "vizwiz_val"
]
current_sheet =  spreadsheet.worksheet("ft_ood_shift")

performance_file_path = "/nethome/bmaneech3/flash/vlm_robustness/result_output/contextual_ood/perf_dict.json"

with open(performance_file_path, 'r') as file : 
    perf_dict = json.load(file)

current_sheet =  spreadsheet.worksheet("FT method shift correlation")
list_of_lists = current_sheet.get_all_values()
corr_last_row = len(list_of_lists)

concept_list = ["image", "joint", "uni_image", "ques_ft", "question"]
for method_idx, ft_method in enumerate(ft_methods) : 

    results_file = f"/coc/pskynet4/bmaneech3/vlm_robustness/result_output/contextual_ood/results_ood/{ft_method}_ood_score_dict.json"
    with open(results_file, 'r') as file : 
        results_dict = json.load(file)

    """for each ft 
    - results_dict : test_split -> concept -> score (int)
        for each split 
        - indiv_results : concept -> instance_id -> score (int)
    """
    current_sheet =  spreadsheet.worksheet("ft_ood_shift")
    list_of_lists = current_sheet.get_all_values()
    last_row = len(list_of_lists)
    current_sheet.update_cell(last_row + 1, 1, ft_method)
    list_of_lists = current_sheet.get_all_values()
    last_row = len(list_of_lists)

    print("ft_method", ft_method)
    cells = [] 
    for concept in concept_list : 
        current_sheet =  spreadsheet.worksheet("ft_ood_shift")
        print(list(results_dict.values()))
        if concept not in list(results_dict.values())[0].keys(): 
            continue 

        list_of_lists = current_sheet.get_all_values()
        last_row = len(list_of_lists)
        current_sheet.update_cell(last_row + 1, 2, concept)
        #find correlation with performance for each concept
        perf_values = [] 
        shift_values = [] 

        for idx, test_split in enumerate(test_splits) : 

            # print("Test split", test_split)
            score = results_dict[test_split][concept]

            if not(ft_method == "vit" or ft_method == "uni_question") : 
                perf_values.append(perf_dict[ft_method][test_split])
            shift_values.append(score)

            cells.append(Cell(row=last_row+1, col=3+idx, value=str(round(score,2))))

            # current_sheet.update_cell(last_row+1, 3+idx, str(round(score,2)))  #round to 4 dp 

        current_sheet.update_cells(cells)

        if ft_method == "vit" or ft_method == "uni_question" : 
            continue
        #calculate correlation score 
        print("Switch to correlation sheet")
        current_sheet =  spreadsheet.worksheet("FT method shift correlation")
        current_sheet.update_cell(corr_last_row + method_idx + 1, 2, ft_method)


        correlation, p_value = pearsonr(shift_values, perf_values)
        col_idx = 3
        if concept == "ques_ft" or concept == "question" : 
            col_idx = 3 

        elif concept == "uni_image" or concept == "image" : 
            col_idx = 4

        else : 
            col_idx = 5 


        current_sheet.update_cell(corr_last_row + method_idx + 1, col_idx, f"{round(correlation, 2)}({round(p_value, 4)})")


        print(f"Correlation for {concept}: {correlation:.2f} (p-value: {p_value:.4f})")
        time.sleep(50)


    

# for ft_method in ft_methods :
#     print("Method :", ft_method)
#     results_file = f"/coc/pskynet4/bmaneech3/vlm_robustness/result_output/contextual_ood/question_ood_score_dict.json"
    
#     with open(results_file, 'r') as file : 
#         results_dict = json.load(file)
#     perf_values = [] 
#     shift_values = []
#     for test_split in test_splits : 
#         shift_score = results_dict[test_split]["question"]
#         acc = perf_dict[ft_method][test_split]
#         perf_values.append(acc)
#         shift_values.append(shift_score)
#     correlation, p_value = pearsonr(shift_values, perf_values)
#     print(f"Correlation for question: {correlation:.2f} (p-value: {p_value:.4f})")




            

# for plot_type in plot_types : 
#     shift_values = [] 
#     perf_values = [] 
#     labels = [] 
#     texts = [] 
#     concepts = [] 
#     print("PLOT_TYPE", plot_type)

#     for data_split, results in combined_ood_perf.items():

#         # if data_split == "vqa_cp_test" or data_split == "vqa_vs_id_val" : 
#         if  data_split == "vqa_vs_id_val" or data_split == "vqa_lol_test" or data_split == "vqa_v2_train": 
#             continue 


#         print(data_split, end = ", ")
        
#         perf_value = perf_dict[data_split]
#         shift_value = results[plot_type]

#         shift_values.append(shift_value)
#         perf_values.append(perf_value)
#         labels.append(f'{data_split}')
        

#     plt.figure(figsize=(10, 6))
#     plt.scatter(shift_values, perf_values, alpha=0.7)

#     for i, label in enumerate(labels):
#         texts.append(
#             plt.text(
#                 shift_values[i], perf_values[i], label,
#                 fontsize=8, ha='center', va='center'
#             )
#         )

#     adjust_text(texts, arrowprops=dict(arrowstyle="-", color='grey', lw=0.5))

#     plt.title(title_dict[plot_type])
#     plt.xlabel('Shift Score')
#     plt.ylabel('Accuracy')
#     plt.grid(True)
#     # plt.legend()
#     plt.savefig(f"/nethome/bmaneech3/flash/vlm_robustness/result_output/ft_{plot_type}_ood_perf.jpg")


#     if len(shift_values) > 1 and len(perf_values) > 1:
#         correlation, p_value = pearsonr(shift_values, perf_values)
#         print(f"Correlation for {plot_type}: {correlation:.2f} (p-value: {p_value:.4f})")
#     else:
#         print(f"Not enough data to calculate correlation for {plot_type}")