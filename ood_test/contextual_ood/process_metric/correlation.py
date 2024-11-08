"""
Possible correlation metrics 
"""
import torch

indiv_dir = "/nethome/bmaneech3/flash/vlm_robustness/result_output/contextual_ood/new_ft_indiv_result"

"""correlation between (Q, J), (I, J)""" 

"""Assume there is some approximate function Joint_shift = f(image_shift, question_shift) -> assume linear first"""

from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression

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




ft_methods = ["pt_emb", "fft", "lora", "lp", "lpft", "ftp", "spd", "digrap"] 

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

concept_list = ["image", "joint", "ques_ft"]
"""for each ft 
    - results_dict : test_split -> concept -> score (int)
        for each split 
        - indiv_results : concept -> instance_id -> score (int)
"""

# concept_pairs = [("image", "joint"), ("ques_ft", "joint")]


"""

for each FT_Method 
"""

current_sheet =  spreadsheet.worksheet("ft_concept_correlation")
# ("uni_image", "image"), 
concept_pairs = [[("image", "joint"), ("ques_ft", "joint")], [("image","image"),("ques_ft","ques_ft"),("joint", "joint")], [("question", "ques_ft")]]

titles = ["Cross concept correlation", "Correlation with PT_VLM", "Correlation with PT_UNIMODAL"]

for title_idx, title in enumerate(titles) : 
    if title_idx == 0 or title_idx == 1 : 
        continue 
    print(f"Running for {title}")
    list_of_lists = current_sheet.get_all_values()
    last_row = len(list_of_lists)
    current_sheet.update_cell(last_row + 1, 1, title)


    for ft_method in ft_methods : 
        #find across all test splits 
        list_of_lists = current_sheet.get_all_values()
        last_row = len(list_of_lists)
        current_sheet.update_cell(last_row + 1, 1, ft_method)

        cells = []
        for split_idx, test_split in enumerate(test_splits) : 
            print("Running for split", test_split)
             
            for concept_idx, (concept_a, concept_b) in enumerate(concept_pairs[title_idx]) : 
            
                first_indiv_res_path = f"/nethome/bmaneech3/flash/vlm_robustness/result_output/contextual_ood/new_ft_indiv_result/{ft_method}/{test_split}_indiv_result.pth"
                first_indiv_res_dict = torch.load(first_indiv_res_path)
                sec_indiv_res_path = f"/nethome/bmaneech3/flash/vlm_robustness/result_output/contextual_ood/new_ft_indiv_result/{ft_method}/{test_split}_indiv_result.pth"
                sec_indiv_res_dict = torch.load(sec_indiv_res_path)

                if title_idx == 1 : 
                    print("Getting PT_EMB")
                    first_indiv_res_path = f"/nethome/bmaneech3/flash/vlm_robustness/result_output/contextual_ood/new_ft_indiv_result/pt_emb/{test_split}_indiv_result.pth"
                    first_indiv_res_dict = torch.load(first_indiv_res_path)

                elif title_idx == 2 : 
                    print("Getting UniModal")
                    if concept_a == "uni_image" : 
                        print("Vit")
                        first_indiv_res_path = f"/nethome/bmaneech3/flash/vlm_robustness/result_output/contextual_ood/new_ft_indiv_result/vit/{test_split}_indiv_result.pth"
                        first_indiv_res_dict = torch.load(first_indiv_res_path)

                    else : 
                        print("BERT uniquestion")
                        first_indiv_res_path = f"/nethome/bmaneech3/flash/vlm_robustness/result_output/contextual_ood/new_ft_indiv_result/uni_question/{test_split}_indiv_result.pth"
                        first_indiv_res_dict = torch.load(first_indiv_res_path)
                        
                concept_a_tensor = [value.to(dtype=torch.float32).item() for key, value in sorted(first_indiv_res_dict[concept_a].items())]
                concept_b_tensor = [value.to(dtype=torch.float32).item() for key, value in sorted(sec_indiv_res_dict[concept_b].items())]
                try : 
                    assert all(elem >= 0 for elem in concept_a_tensor) and all(elem >= 0 for elem in concept_b_tensor) , "found negative shift values"
                except AssertionError as e : 
                    print(e)
                    continue 
        
                correlation, p_value = pearsonr(concept_a_tensor, concept_b_tensor)
                #write to google sheets 
                
                current_sheet.update_cell(last_row + 1 + concept_idx, 2, str(f"({concept_a},{concept_b})"))

                cells.append(Cell(row=last_row+1 + concept_idx, col=3+split_idx, value=f"{round(correlation, 2)}({round(p_value, 4)})"))
                print(f"Correlation for {concept_a}, {concept_b}: {correlation:.2f} (p-value: {p_value:.4f})")      


        current_sheet.update_cells(cells)










                





            














