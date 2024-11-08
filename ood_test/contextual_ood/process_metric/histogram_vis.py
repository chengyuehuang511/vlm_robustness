import matplotlib.pyplot as plt
from PIL import Image

import torch
import json 
import gspread
import time 
from gspread import Cell 
import os
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from oauth2client.service_account import ServiceAccountCredentials
import numpy as np 
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import argparse
import logging 
import random
from datetime import datetime
random.seed(42)

#TODO set seed



scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]


creds = ServiceAccountCredentials.from_json_keyfile_name('/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/credential.json', scope)

client = gspread.authorize(creds)

gauth = GoogleAuth()
gauth.credentials = creds  # Use credentials from gspread if already authenticated
drive = GoogleDrive(gauth)

# spreadsheet = client.open_by_key('1E5JWHC42e8emwtoCsyLRBUBlI9AtMz0uBpID2wX9Gko')

SERVICE_ACCOUNT_FILE = '/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/credential.json'
SCOPES = ['https://www.googleapis.com/auth/drive.file']
credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES)
service = build('drive', 'v3', credentials=credentials)

ds_2_img = json.load(open("/coc/pskynet4/bmaneech3/vlm_robustness/ood_test/other/ds_2_img.json"))

ds_split_2_file = json.load(open("/coc/pskynet4/bmaneech3/vlm_robustness/ood_test/other/ds_split_2_file.json"))

title_map = {"image" : "Visual", 
             "joint" : "V+Q",
             "question" : "Question", 
             "ques_ft" : "Question Finetuned",
             "uni_image" : "Visual"
            }

"""for each ft : 
    - results_dict (dict) : test_split -> concept -> score (int)
    for each split : 
        - indiv_results (dict) : concept -> instance_id -> score (int)
"""

"""
1) get_intersect_samples()
2) get_left_region() 
3) get_top_region() 
4) get_right_region() 
5) image_path_to_url()


stuff you want to do : 
- store region ranges for each ft_method and concept 
    - ft_method -> concept -> (st, end) range 

- 
"""


folder_id = "1VAcC-Q_wwd8vX1pmg8Fjs8xog9mdynnM"


def get_spreadsheet(file_name) : 
    """
    args : 
        file_name (str) 

    return : 
        spreadsheet (object)    
    """

    #check if file exists : 
    file_list = drive.ListFile({'q': f"title = '{file_name}' and mimeType = 'application/vnd.google-apps.spreadsheet'"}).GetList()

    #if file exists retrieve file id 
    if file_list:
        file = file_list[0]
        file_id = file['id']
        file_link = f"https://docs.google.com/spreadsheets/d/{file_id}"
        logging.info(f"File found: {file_name}, Link: {file_link}")

    else:
        new_file = drive.CreateFile({
            'title': file_name,
            'mimeType': 'application/vnd.google-apps.spreadsheet',
            'parents': [{'id': folder_id}]
        })
        new_file.Upload()
        file_id = new_file['id']
        file_link = f"https://docs.google.com/spreadsheets/d/{file_id}"
        logging.info(f"Spreadsheet created: {file_name}, Link: {file_link}")


    spreadsheet = client.open_by_key(file_id)

    return spreadsheet 


def get_left_tail(test_arr, hist, bin_edges, percentile=0.01) : 
    """
    arg : 
        test_arr (tensor) : (n, 1) -> in order of instance_id
        hist : (object)
        bin_edges 

    return : 
        - region range (a,b)
        - list of indices 
    """

    counts = torch.cumsum(hist, dim=0)
    total_elements = counts[-1].item()

    threshold = total_elements * percentile
    
    bool_mask = counts >= threshold 
    int_tensor = bool_mask.int()

    left_tail_bin_index = torch.argmax(int_tensor).item()

    # Range corresponding to the left percentile%
    left_tail_range = (bin_edges[0].item(), bin_edges[left_tail_bin_index + 1].item())

    #get indices from that range 
    start, end = left_tail_range

    mask = (test_arr >= start) & (test_arr <= end)
    indices = torch.nonzero(mask).squeeze()

    return (start, end), indices 

def get_right_tail(test_arr, hist, bin_edges, percentile=0.99) : 

    """
    arg : 
        test_arr (tensor) : (n, 1) -> in order of instance_id
        hist : (object)
        bin_edges 

    return : 
        - region range (a,b)
        - list of indices 
    """

    counts = torch.cumsum(hist, dim=0)
    print("count dtype", counts.dtype)
    total_elements = counts[-1].item()


    #99th percentile : 1/100 = 0.01
    threshold = total_elements * percentile
 
    bool_mask = counts >= threshold
    int_tensor = bool_mask.int()

    right_tail_bin_index = torch.argmax(int_tensor).item()

    # Range corresponding to the top 1% of the right 
    right_tail_range = (bin_edges[0].item(), bin_edges[right_tail_bin_index + 1].item())

    #get indices from that range 
    start, end = right_tail_range

    # mask = (arr1 >= start) & (arr1 <= end) #fix code 
    mask = (test_arr >= end)
    indices = torch.nonzero(mask).squeeze()

    return  (start, end), indices

def get_top_samples(test_arr, hist, bin_edges) : 

    top_bins_idx = torch.topk(hist, 1).indices #(n, )
    num_elems =  top_bins_idx.size(0) 

    # Extract the range for each of the top bins
    top_bin_ranges = [(bin_edges[i].item(), bin_edges[i + 1].item()) for i in top_bins_idx]

    start, end = top_bin_ranges[0]

    start -= 1
    end += 1

    #verify bin ranges 
    logging.info(f"Bin ranges {top_bin_ranges} : ({start}, {end})")

    mask = (test_arr >= start) & (test_arr <=end) #find mask positions 
    indices = torch.nonzero(mask).squeeze()

    #also annotate start end on hist 
    return (start, end), indices

def get_intersect_samples(train_arr, test_arr, train_hist, test_hist, bin_edges): 
    intersection = np.minimum(train_hist, test_hist)
    idx = np.argmax(intersection)
    print(len(intersection))
    start, end = bin_edges[idx], bin_edges[idx+1]
    print("intersect range before shift", start, end)

    start -= 0.5
    end += 0.5

    mask_train = (train_arr >= start) & (train_arr <= end)
    train_indices = torch.nonzero(mask_train).squeeze()
    mask_test= (test_arr >= start) & (test_arr <= end)
    test_indices = torch.nonzero(mask_test).squeeze()

    # train_inter_samples, test_inter_samples, inter_range = get_intersect_samples(train_samples, test_samples, train_hist, test_hist, bin_edges)

    return train_indices, test_indices, (start,end)

def retrieve_samples(raw_data, indices): 
    """
    retrieve the samples at given indices and return needed attributes 

    args : 
        - raw_data : combined ann list  
    
    return : 
        - list of selected samples at given indices
    """
    
    return [raw_data[i] for i in indices]

def image_path_to_url(image_split, file_name, folder_id) : 
    image_path = os.path.join(image_split, file_name)
    file_name = file_name[file_name.find('/')+1:]
    file_metadata = {
    'name': file_name,
    'parents': [folder_id]  # Specify the folder ID
    }

    media = MediaFileUpload(image_path, mimetype='image/jpeg')
    file = service.files().create(
        body=file_metadata,
        media_body=media,
        fields='id'
    ).execute()
    file_id = file.get('id')
    print(f'File ID: {file_id}')

    file_link = f'https://drive.google.com/uc?id={file_id}'
    print("File link", file_link)
    return file_link



test_splits = [

    "coco_vqav2_train_val",
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



"""for each ft 
        - results_dict (dict) : test_split -> concept -> score (int)
        for each split
            - indiv_results (dict) : concept -> instance_id -> score (int)
"""
region_list = ["left", "top", "right", "intersect"]
if __name__ == "__main__" : 

    #For each ft_method -> test split -> concept 

    parser = argparse.ArgumentParser(description="Process a parameter.")
    parser.add_argument("--ft_method", type=str, required=True, help="The parameter to be processed")
    args = parser.parse_args()
    ft_method = args.ft_method

    log_output_dir = "/nethome/bmaneech3/flash/vlm_robustness/result_output/logs/"
    os.makedirs(os.path.join(log_output_dir, ft_method), exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_filename = f"hist_vis_{timestamp}.log"
    log_file = os.path.join(log_output_dir, ft_method, log_filename)
    
    logging.basicConfig(
    level=logging.INFO,
    format="%(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file),  # Log to file
        logging.StreamHandler()         # Log to console
    ]
    )

    """
    range_dict : 
    for each ft_method 
        - test_split -> concept -> region_type -> (a,b) range
    """
    range_dir = "/nethome/bmaneech3/flash/vlm_robustness/result_output/contextual_ood/threshold"


    #TODO add logging
    train_split = "coco_vqav2_train_val"

    name_map = json.load(open("/nethome/bmaneech3/flash/vlm_robustness/ood_test/other/name_map.json"))
    ds_2_ann = json.load(open("/nethome/bmaneech3/flash/vlm_robustness/ood_test/other/ds_split_2_file.json"))
    ds_2_img = json.load(open("/nethome/bmaneech3/flash/vlm_robustness/ood_test/other/ds_2_img.json"))

    diagram_dir = "/nethome/bmaneech3/flash/vlm_robustness/result_output/contextual_ood/diagrams/histogram"
    #store with and without annotations

    train_file_path =   f"/nethome/bmaneech3/flash/vlm_robustness/result_output/contextual_ood/new_ft_indiv_result/{ft_method}/{train_split}_indiv_result.pth"
    train_dict = torch.load(train_file_path) 
    #num_bins 
    num_bins = 100 

    spreadsheet = get_spreadsheet(ft_method)

    for test_split in test_splits : 
        range_dict_file = os.path.join(range_dir, f"{ft_method}.json")
        if os.path.exists(range_dict_file) : 
            with open(range_dict_file, 'r') as file : 
                range_dict = json.load(file)
        else : 
            range_dict = {} 


        if test_split in range_dict : 
            continue 
        range_dict[test_split] = {} 


        test_file_path = f"/nethome/bmaneech3/flash/vlm_robustness/result_output/contextual_ood/new_ft_indiv_result/{ft_method}/{test_split}_indiv_result.pth"
        test_dict = torch.load(test_file_path) 

        for concept in test_dict : 
            cur_train_dict = train_dict[concept]
            cur_test_dict = test_dict[concept] #{instance_id : score} 
            sorted_train_tensor = torch.tensor([value for _,value in sorted(cur_train_dict.items())]).to(dtype=torch.float32)
            sorted_test_tensor = torch.tensor([value for _,value in sorted(cur_test_dict.items())]).to(dtype=torch.float32) #(batch size)

            #left, top, right, intersect 

            range_label = {} 
            train_list, train_bin_edges, _ = plt.hist(sorted_train_tensor, density=True, bins=num_bins, alpha=0.5, label=f'{name_map[train_split]}')
            test_list, test_bin_edges, _ = plt.hist(sorted_test_tensor, density=True, bins=num_bins, alpha=0.5, label=f'{name_map[test_split]}')
            
            train_list = torch.tensor(train_list)
            test_list = torch.tensor(test_list)

            for region_type in region_list : 

                if region_type == "intersect" :     
                    combined_min = min(train_bin_edges.min(), test_bin_edges.min())
                    combined_max = max(train_bin_edges.max(), test_bin_edges.max())

                    combined_bin_edges = np.linspace(combined_min, combined_max, num_bins + 1)
                    train_idx_samples, test_idx_samples, region_range = get_intersect_samples(sorted_train_tensor, sorted_test_tensor, train_list, test_list, combined_bin_edges)

                
                elif region_type == "left" : 
                    region_range, test_idx_samples = get_left_tail(sorted_test_tensor, test_list, test_bin_edges)
                elif region_type == "right" : 
                    region_range, test_idx_samples = get_right_tail(sorted_test_tensor, test_list, test_bin_edges)

                elif region_type == "top" : 
                    region_range, test_idx_samples = get_top_samples(sorted_test_tensor, test_list, test_bin_edges)

                range_label[region_type] = region_range

                #randomly sample 
                num_elements = 50 

                if region_type == "intersect" :     
                    train_idx_samples = train_idx_samples.numpy()
                    train_selected_idxs = np.random.choice(train_idx_samples, num_elements, replace=False)
                
                    cur_combined_ann = json.load(open(ds_2_ann[name_map[train_split]]))
                    train_selected_samples = retrieve_samples(cur_combined_ann, train_selected_idxs)


                test_idx_samples = test_idx_samples.numpy()
                test_selected_idxs = np.random.choice(test_idx_samples, num_elements, replace=False)

                cur_combined_ann = json.load(open(ds_2_ann[name_map[test_split]]))

                test_selected_samples = retrieve_samples(cur_combined_ann, test_selected_idxs)

                logging.info(f"Running for {test_split}, {concept},{region_type}]")

                #create an image folder : {concept}_{inspect_type}_{ds_name}
                folder_metadata = {
                    'name': f"{ft_method}_{region_type}_{concept}_{test_split}",
                    'mimeType': 'application/vnd.google-apps.folder'
                }
                folder = service.files().create(
                    body=folder_metadata,
                    fields='id'
                ).execute()
                print("folder link", )
                folder_id = folder.get('id')

                permission = {
                    'type': 'anyone',
                    'role': 'reader'
                }
                service.permissions().create(
                    fileId=folder_id,
                    body=permission
                ).execute()
                logging.info('Folder permissions set to public.')
                folder_link = f'https://drive.google.com/drive/folders/{folder_id}'
                logging.info(f"Folder_link {folder_link}")

                sheet_name = f"{concept}_{region_type}"

                #create new worksheet if doesn't exist 
                try:
                    worksheet = spreadsheet.worksheet(sheet_name)
                    logging.info(f"Sheet '{sheet_name}' already exists.")
                except gspread.exceptions.WorksheetNotFound:
                    worksheet = spreadsheet.add_worksheet(title=sheet_name, rows="10000", cols="20")
                    logging.info(f"Sheet '{sheet_name}' created successfully.")

                #update results 
                cells = [] 

                list_of_lists = worksheet.get_all_values()
                last_row = len(list_of_lists)

                worksheet.update_cell(last_row + 1, 5, folder_link)
                worksheet.update_cell(last_row + 1, 1, str(region_range))
                list_of_lists = worksheet.get_all_values()
                last_row = len(list_of_lists)

                if region_type == "intersect" : 
                    co = 0 
                    for i, sample in enumerate(train_selected_samples, start=last_row+1):
                        time.sleep(1)
                        question = sample['question']
                        answers = ", ".join(sample['answer'])  # Join answers as a single string
                        image_url = image_path_to_url(ds_2_img[name_map[test_split]], sample['image'], folder_id)  # Convert image path to URL

                        cells.append(Cell(row=i, col=1, value=test_split))
                        cells.append(Cell(row=i, col=2, value=sample['question_id']))  # Column A: question_id
                        cells.append(Cell(row=i, col=3, value=question))             # Column B: question
                        cells.append(Cell(row=i, col=4, value=answers))             # Column C: answers
                        cells.append(Cell(row=i, col=5, value=image_url))             # Column D: image URL
                        cells.append(Cell(row=i, col=6, value=str(cur_train_dict[train_selected_idxs[i]].item()))) #maha score
                        co+= 1 

                    worksheet.update_cells(cells)
                    list_of_lists = worksheet.get_all_values()
                    last_row = len(list_of_lists)

                co = 0 
                for i, sample in enumerate(test_selected_samples, start=last_row+1):
                    time.sleep(1)
                    question = sample['question']
                    answers = ", ".join(sample['answer'])  # Join answers as a single string
                    image_url = image_path_to_url(ds_2_img[name_map[test_split]], sample['image'], folder_id)  # Convert image path to URL

                    cells.append(Cell(row=i, col=1, value=test_split))
                    cells.append(Cell(row=i, col=2, value=sample['question_id']))  # Column A: question_id
                    cells.append(Cell(row=i, col=3, value=question))             # Column B: question
                    cells.append(Cell(row=i, col=4, value=answers))             # Column C: answers
                    cells.append(Cell(row=i, col=5, value=image_url))             # Column D: image URL
                    cells.append(Cell(row=i, col=6, value=str(cur_test_dict[test_selected_idxs[i]].item()))) #maha score
                    co+= 1 

                worksheet.update_cells(cells)

                list_of_lists = worksheet.get_all_values()
                last_row = len(list_of_lists)
                worksheet.update_cell(last_row + 1, 1, "Done ==========")


            plt.title(f'Histogram {title_map[concept]}  - {name_map[test_split]}')
            
            plt.xlabel('Mahalanobis score')
            plt.ylabel('Frequency')
            plt.legend()
            
            plt.savefig(os.path.join(diagram_dir, "without_ann", f"{ft_method}_{concept}_{test_split}.jpg"))

            colors = ["green", "red", "yellow", "purple"] 
            logging.info(range_label)

            #annotate regions 
            for i, (label_name, (start, end)) in enumerate(range_label.items()): 
                if label_name == "right_tail" or label_name == "left_tail": 
                    plt.axvline(x=end,linestyle='--', color=colors[i], alpha=0.7)

                    # plt.axvline(x=start, linestyle='--', color=colors[i], alpha=0.7)

                else : 

                    plt.axvline(x=start, linestyle='--', color=colors[i], alpha=0.7)
                    plt.axvline(x=end,linestyle='--', color=colors[i], alpha=0.7)


            plt.text(x=0.95, y=0.95, s='Green = Top\nRed = Intersect\nYellow = Left Tail\nPurple = Right Tail',
            horizontalalignment='right', verticalalignment='top',
            transform=plt.gca().transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
            

            plt.legend()
            plt.savefig(os.path.join(diagram_dir, "with_ann", f"{ft_method}_{concept}_{test_split}.jpg"))

            
            # plt.savefig(f"/nethome/bmaneech3/flash/vlm_robustness/result_output/contextual_ood/new_hist_plot/v+q+a/no_annot/f_annot_{hidden_layer_name[c]}_{train_ds_split}_{test_ds_split}.jpg")
            plt.close()

            time.sleep(30)

            range_dict[test_split][concept] = range_label

        with open(range_dict_file, range_dict) as file : 
            json.dump(range_dict, file)

