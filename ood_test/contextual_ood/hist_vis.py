import matplotlib.pyplot as plt

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
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]

creds = ServiceAccountCredentials.from_json_keyfile_name('/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/credential.json', scope)

client = gspread.authorize(creds)

# sheetTop = client.open_by_key('1Xt4P21X9I9tH_u5H-EHTzeMeM5X-oRCyAeQDtTSg-oU').sheet1  # sheet1 refers to the first sheet

# sheetIntersect =  client.open_by_key('1Xt4P21X9I9tH_u5H-EHTzeMeM5X-oRCyAeQDtTSg-oU').sheet2
# sheetLeft =  client.open_by_key('1Xt4P21X9I9tH_u5H-EHTzeMeM5X-oRCyAeQDtTSg-oU').sheet3

spreadsheet = client.open_by_key('1Xt4P21X9I9tH_u5H-EHTzeMeM5X-oRCyAeQDtTSg-oU')
spreadsheet = client.open_by_key('1Xt4P21X9I9tH_u5H-EHTzeMeM5X-oRCyAeQDtTSg-oU')



SERVICE_ACCOUNT_FILE = '/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/credential.json'
SCOPES = ['https://www.googleapis.com/auth/drive.file']
credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES)
service = build('drive', 'v3', credentials=credentials)

torch.manual_seed(0)
tensor = torch.randn(2, 1999) 

ds_2_img = { 
    "advqa" : "/srv/datasets/coco/", 
    "cvvqa" : "/coc/pskynet4/chuang475/projects/vlm_robustness/tmp/datasets/cv-vqa/val/BS/vedika2/nobackup/thesis/IMAGES_counting_del1_edited_VQA_v2/",
    "vqa_v2" : "/coc/pskynet6/chuang475/.cache/lavis/coco/images/", 
    "ivvqa": "/coc/pskynet4/chuang475/projects/vlm_robustness/tmp/datasets/iv-vqa/val/BS/vedika2/nobackup/thesis/final_edited_VQA_v2/Images/", 
    "okvqa" : "/srv/datasets/coco/",
    "textvqa" : "/coc/pskynet4/chuang475/projects/vlm_robustness/tmp/datasets/textvqa/", 
    "vizwiz" : "/srv/datasets/vizwiz/data/Images/", 
    "vqa_ce" : "/coc/pskynet6/chuang475/.cache/lavis/coco/images/", 
    "vqa_cp" : "/coc/pskynet6/chuang475/.cache/lavis/coco/images/", 
    "vqa_lol" : "/coc/pskynet6/chuang475/.cache/lavis/coco/images/", 
    "vqa_rephrasings" : "/coc/pskynet6/chuang475/.cache/lavis/coco/images/",
    "vqa_vs" : "/coc/pskynet6/chuang475/.cache/lavis/coco/images/"
}

ds_split_2_file = { 
    "vqa_v2_train" : "/coc/pskynet6/chuang475/.cache/lavis/coco/annotations/vqa_train.json",
    "vqa_v2_val" : "/coc/pskynet6/chuang475/.cache/lavis/coco/annotations/vqa_val_eval.json", 
    "vqa_v2_test": "/coc/pskynet6/chuang475/.cache/lavis/coco/annotations/vqa_test.json" , 
    "advqa_test" : "/coc/pskynet4/chuang475/projects/vlm_robustness/tmp/datasets/advqa/val/combined_data.json", 
    "cvvqa_test" : "/coc/pskynet4/chuang475/projects/vlm_robustness/tmp/datasets/cv-vqa/val/BS/vedika2/nobackup/thesis/mini_datasets_qa_COUNTING_DEL_1/0.1_0.0/combined_data.json", 
    "ivvqa_test" : "/coc/pskynet4/chuang475/projects/vlm_robustness/tmp/datasets/iv-vqa/val/BS/vedika2/nobackup/thesis/mini_datasets_qa/0.1_0.1/combined_data.json",
    "okvqa_test" : "/coc/pskynet4/chuang475/projects/vlm_robustness/tmp/datasets/ok-vqa/val/combined_data.json",
    "textvqa_test" : "/coc/pskynet4/chuang475/projects/vlm_robustness/tmp/datasets/textvqa/val/combined_data.json",
    "textvqa_train" : "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/textvqa/train/combined_data.json", 
    "vizwiz_test" :  "/coc/pskynet4/chuang475/projects/vlm_robustness/tmp/datasets/vizwiz/val/combined_data.json",
    "vqa_ce_test" : "/coc/pskynet4/chuang475/projects/vlm_robustness/tmp/datasets/vqace/val/combined_data_subset.json",
    "vqa_cp_train" : "/coc/pskynet4/chuang475/projects/vlm_robustness/tmp/datasets/vqacp2/train/vqacp_v2_train_questions.json", 
    "vqa_cp_test" : "/coc/pskynet4/chuang475/projects/vlm_robustness/tmp/datasets/vqacp2/test/combined_data.json", 
    "vqa_lol_train": "/coc/pskynet4/bmaneech3/vlm_robustness/tmp/datasets/vqalol/train/combined_data.json",
    "vqa_lol_test": "/coc/pskynet4/bmaneech3/vlm_robustness/tmp/datasets/vqalol/test/combined_data.json", 
    "vqa_rephrasings_test": "/coc/pskynet4/chuang475/projects/vlm_robustness/tmp/datasets/vqa_rephrasings/combined_data.json",
    "vqa_vs_train" : "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/train/combined_data.json", 
    "vqa_vs_id_val" : "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/val/combined_data.json", 
    "vqa_vs_id_test" : "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/combined_data.json", 
    "vqa_vs_ood_test" : "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/combined_data.json", 
    "vqa_vs_KO" : "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/KO/combined_data.json",
    "vqa_vs_KOP" : "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/KOP/combined_data.json", 
    "vqa_vs_KW" : "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/KW/combined_data.json", 
    "vqa_vs_KW_KO" : "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/KW+KO/combined_data.json", 
    "vqa_vs_KWP" : "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/KWP/combined_data.json", 
    "vqa_vs_QT" : "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/QT/combined_data.json", 
    "vqa_vs_QT_KO" : "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/QT+KO/combined_data.json", 
    "vqa_vs_QT_KW" : "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/QT+KW/combined_data.json", 
    "vqa_vs_QT_KW_KO" : "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/QT+KW+KO/combined_data.json"
}

hidden_layer_name = ["question_mid", "question_last"]


def get_top_samples(arr1, hist, bin_edges, k = 50) : 

    top_bins_idx = torch.topk(hist, 1).indices #(n, )
    num_elems =  top_bins_idx.size(0) 

    # Extract the range for each of the top bins
    top_bin_ranges = [(bin_edges[i].item(), bin_edges[i + 1].item()) for i in top_bins_idx]

    start, end = top_bin_ranges[0]

    start -= 1.5
    end += 1.5
   

    #verify bin ranges 
    print(f"Bin ranges {top_bin_ranges} : ({start}, {end})")

    mask = (arr1 >= start) & (arr1 <=end) #find mask positions 
    indices = torch.nonzero(mask).squeeze()

    #also annotate start end on hist 
    return indices, (start, end)


def get_intersect_samples(arr1, arr2, hist1, hist2, bin_edges_1): 
    intersection = torch.min(hist1, hist2)
    best_inter_idx = torch.argmax(intersection).item()
    print("type best intersect idx", best_inter_idx)


    intersect_ranges = [(bin_edges_1[best_inter_idx].item(), bin_edges_1[best_inter_idx + 1].item())]
    
    
    start, end = intersect_ranges[0]
    print("intersect range before shift", start, end)

    start -= 1.5
    end += 1.5

    mask = (arr1 >= start) & (arr1 <= end)
    train_indices = torch.nonzero(mask).squeeze()
    mask = (arr2 >= start) & (arr2 <= end)
    test_indices = torch.nonzero(mask).squeeze()
    

    return train_indices, test_indices,  (start,end)



def get_left_tail(arr1, hist, bin_edges) : 
    # hist = torch.histc(arr1, bins=num_bins, min=arr1.min().item(), max=arr1.max().item())

    # bin_edges = torch.linspace(arr1.min().item(), arr1.max().item(), num_bins + 1)
    counts = torch.cumsum(hist, dim=0)
    print("count dtype", counts.dtype)
    total_elements = counts[-1].item()


    #25th percentile 
    threshold = total_elements * 0.25

    bool_mask = counts >= threshold
    int_tensor = bool_mask.int()

    left_tail_bin_index = torch.argmax(int_tensor).item()

    # Range corresponding to the left 25%
    left_tail_range = (bin_edges[0].item(), bin_edges[left_tail_bin_index + 1].item())

    #get indices from that range 
    start, end = left_tail_range

    mask = (arr1 >= start) & (arr1 <= end)
    indices = torch.nonzero(mask).squeeze()

    return  indices,  left_tail_range

def retrieve_samples(raw_data, vis_path ,indices): 
    """
    retrieve the samples at given indices and return needed attributes 
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



#dataset combo : 
splits =[
    #ds_name, split, file_name
    #(train, test, concept)
    #vqav2 train with all others
    #train_stuff = sample[0], test_stuff = sample[1]
    [("vqa_v2","train"), ("vqa_v2","train")],
    [("vqa_v2","train"), ("vqa_v2","val")], 
    [("vqa_v2","train"), ("advqa", "test")],
    [("vqa_v2","train"), ("cvvqa", "test")], 
    [("vqa_v2","train"), ("ivvqa", "test")],
    [("vqa_v2","train"), ("okvqa", "test")], 
    [("vqa_v2","train"), ("textvqa", "test")], 
    [("vqa_v2","train"), ("vizwiz", "test")], 
    [("vqa_v2","train"), ("vqa_ce", "test")], 
    [("vqa_v2","train"), ("vqa_rephrasings", "test")],
    # [("vqa_v2","train"), ("vqa_vs", "id_val")], 
    # [("vqa_v2","train"), ("vqa_v2","test")], 
    [("vqa_v2", "train"), ("vqa_cp", "test")]
    # [("vqa_v2", "train"), ("vqa_lol", "test")]
]

n = 3 

concept_type = ["question"] #["question"]


for concept in concept_type : 

    for split in splits : 
        print("Running", split)
        train_split = split[0]
        train_ds_name, train_split = split[0]
        test_ds_name, test_split = split[1]

        train_ds_split = f"{train_ds_name}_{train_split}" 
        test_ds_split = f"{test_ds_name}_{test_split}"

        train_vectors = torch.load(f"/coc/pskynet4/bmaneech3/vlm_robustness/result_output/contextual_ood/indiv_result/{train_ds_split}_{concept}.pth")
        test_vectors = torch.load(f"/coc/pskynet4/bmaneech3/vlm_robustness/result_output/contextual_ood/indiv_result/{test_ds_split}_{concept}.pth")
        if train_vectors.dtype == torch.bfloat16: 
            train_vectors = train_vectors.to(dtype=torch.float32)
            torch.save(train_vectors, f"/coc/pskynet4/bmaneech3/vlm_robustness/result_output/contextual_ood/indiv_result/{train_ds_split}_{concept}.pth")
        if test_vectors.dtype == torch.bfloat16:
            test_vectors = test_vectors.to(dtype=torch.float32)
            torch.save(test_vectors, f"/coc/pskynet4/bmaneech3/vlm_robustness/result_output/contextual_ood/indiv_result/{test_ds_split}_{concept}.pth")

        dataset_labels = [train_ds_name, test_ds_name] 

        datasets = [train_vectors, test_vectors]  
        

        for c in range(datasets[0].shape[0]):  
            #c = current concept type 
            plt.figure(figsize=(10, 6))
            train_select_samples = None 
            test_select_samples = None
            train_inter_samples = None 
            test_inter_samples = None 
            test_left_samples = None
            #top, intersect, tail 
            for i, dataset in enumerate(datasets):
                plt.hist(dataset[c].numpy(), density=True, bins=50, alpha=0.5, label=f'{dataset_labels[i]} (c={c})')
 
            num_bins = 50 
            train_samples = datasets[0][c]
            test_samples = datasets[1][c]
            mi = min(train_samples.min().item(), test_samples.min().item()) 
            ma = min(train_samples.max().item(), test_samples.max().item())
            train_hist = torch.histc(train_samples, bins=num_bins, min=mi, max=ma)
            test_hist = torch.histc(test_samples, bins=num_bins, min=mi, max=ma)
            # Bin edges
            bin_edges = torch.linspace(mi, ma , num_bins + 1)
            #sample structure : [list of indices]
            train_top_samples, train_top_range =  get_top_samples(train_samples, train_hist, bin_edges)
            test_top_samples, test_top_range = get_top_samples(test_samples, test_hist, bin_edges)
            train_inter_samples, test_inter_samples, inter_range = get_intersect_samples(train_samples, test_samples, train_hist, test_hist, bin_edges)
            train_left_samples, train_left_range = get_left_tail(train_samples, train_hist, bin_edges)
            test_left_samples, test_left_range = get_left_tail(test_samples, test_hist, bin_edges)


            with open(ds_split_2_file[train_ds_split], 'r') as file : 
                train_raw_data = json.load(file)
            with open(ds_split_2_file[test_ds_split], 'r') as file : 
                test_raw_data = json.load(file)



            #randomly select 50 samples
            #top samples 
            print(f"Running for {hidden_layer_name[c]}_top, {test_ds_split}")

            sheet_name = f"{hidden_layer_name[c]}_top"
            k = 50
            random_idxs = torch.randint(low=0, high=train_top_samples.size(0), size=(k,))
            train_top_samples = train_top_samples[random_idxs]
            new_train_top_samples = [train_raw_data[i] for i in train_top_samples]

            random_idxs = torch.randint(low=0, high=test_top_samples.size(0), size=(k,))
            test_top_samples = test_top_samples[random_idxs]
            new_test_top_samples = [test_raw_data[i] for i in test_top_samples]

            #create an image folder : {concept}_{inspect_type}_{ds_name}
            folder_metadata = {
                'name': f"{hidden_layer_name[c]}_top_{test_ds_split}",
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
            print('Folder permissions set to public.')


            folder_link = f'https://drive.google.com/drive/folders/{folder_id}'
            print("Folder_link", folder_link)
            current_sheet =  spreadsheet.worksheet(sheet_name)
            list_of_lists = current_sheet.get_all_values()
            last_row = len(list_of_lists)
            current_sheet.update_cell(last_row + 1, 5, folder_link)
            list_of_lists = current_sheet.get_all_values()
            last_row = len(list_of_lists)
            cells = [] 
            for i, sample in enumerate(new_test_top_samples, start=last_row+1):
                time.sleep(1)
                question = sample['question']
                answers = ", ".join(sample['answer'])  # Join answers as a single string
                image_url = image_path_to_url(ds_2_img[test_ds_name], sample['image'], folder_id)  # Convert image path to URL

                cells.append(Cell(row=i, col=1, value=test_ds_split))
                cells.append(Cell(row=i, col=2, value=sample['question_id']))  # Column A: question_id
                cells.append(Cell(row=i, col=3, value=question))             # Column B: question
                cells.append(Cell(row=i, col=4, value=answers))             # Column C: answers
                cells.append(Cell(row=i, col=5, value=image_url))             # Column D: image URL

                # current_sheet.update_cell(i, 1, test_ds_split)
                # current_sheet.update_cell(i, 2, sample['question_id'])  # Column A: question_id
                # current_sheet.update_cell(i, 3, question)              # Column B: question
                # current_sheet.update_cell(i, 4, answers)               # Column C: answers
                # current_sheet.update_cell(i, 5, image_url)             # Column D: image URL
            current_sheet.update_cells(cells)

#==========================================================================================
            #left tail 

            print(f"Running for {hidden_layer_name[c]} left tail, {test_ds_split}")
            sheet_name = f"{hidden_layer_name[c]}_left"
            random_idxs = torch.randint(low=0, high=test_left_samples.size(0), size=(k,))
            test_left_samples = test_left_samples[random_idxs]
            new_test_left_samples = [test_raw_data[i] for i in test_left_samples]

            #create an image folder : {concept}_{inspect_type}_{ds_name}
            folder_metadata = {
                'name': f"{hidden_layer_name[c]}_left_{test_ds_split}",
                'mimeType': 'application/vnd.google-apps.folder'
            }
            folder = service.files().create(
                body=folder_metadata,
                fields='id'
            ).execute()
            folder_id = folder.get('id')

            permission = {
                'type': 'anyone',
                'role': 'reader'
            }
            service.permissions().create(
                fileId=folder_id,
                body=permission
            ).execute()
            print('Folder permissions set to public.')

            folder_link = f'https://drive.google.com/drive/folders/{folder_id}'
            print("Folder_link", folder_link)

            cells = [ ]
            current_sheet =  spreadsheet.worksheet(sheet_name)
            list_of_lists = current_sheet.get_all_values()
            last_row = len(list_of_lists)
            current_sheet.update_cell(last_row + 1, 5, folder_link)
            list_of_lists = current_sheet.get_all_values()
            last_row = len(list_of_lists)
            for i, sample in enumerate(new_test_left_samples, start=last_row+1):
                question = sample['question']
                answers = ", ".join(sample['answer'])  # Join answers as a single string
                print(test_ds_name)
                image_url = image_path_to_url(ds_2_img[test_ds_name], sample['image'], folder_id)  # Convert image path to URL

                cells.append(Cell(row=i, col=1, value=test_ds_split))
                cells.append(Cell(row=i, col=2, value=sample['question_id']))  # Column A: question_id
                cells.append(Cell(row=i, col=3, value=question))             # Column B: question
                cells.append(Cell(row=i, col=4, value=answers))             # Column C: answers
                cells.append(Cell(row=i, col=5, value=image_url))             # Column D: image URL

            current_sheet.update_cells(cells)
            
#============================================================================================================
            #intersect 
            
            print(f"Running for {hidden_layer_name[c]} intersect, {train_ds_split}")                              
            sheet_name = f"{hidden_layer_name[c]}_intersect"

            random_idxs = torch.randint(low=0, high=train_inter_samples.size(0), size=(k,))
            train_inter_samples = train_inter_samples[random_idxs]
            new_train_inter_samples = [train_raw_data[i] for i in train_inter_samples]

            random_idxs = torch.randint(low=0, high=test_inter_samples.size(0), size=(k,))
            test_inter_samples = test_inter_samples[random_idxs]
            new_test_inter_samples = [test_raw_data[i] for i in test_inter_samples]

            #create an image folder : {concept}_{inspect_type}_{ds_name}
            folder_metadata = {
                'name': f"{hidden_layer_name[c]}_intersect_{train_ds_split}_{test_ds_split}",
                'mimeType': 'application/vnd.google-apps.folder'
            }
            folder = service.files().create(
                body=folder_metadata,
                fields='id'
            ).execute()
            folder_id = folder.get('id')

            permission = {
                'type': 'anyone',
                'role': 'reader'
            }
            service.permissions().create(
                fileId=folder_id,
                body=permission
            ).execute()
            print('Folder permissions set to public.')

            folder_link = f'https://drive.google.com/drive/folders/{folder_id}'
            print("Folder_link", folder_link)
            current_sheet =  spreadsheet.worksheet(sheet_name)

            list_of_lists = current_sheet.get_all_values()
            last_row = len(list_of_lists)
            current_sheet.update_cell(last_row + 1, 5, folder_link)
            list_of_lists = current_sheet.get_all_values()
            last_row = len(list_of_lists)
            cells = [] 
            for i, sample in enumerate(new_train_inter_samples, start=last_row+1):
                question = sample['question']
                answers = ", ".join(sample['answer'])  # Join answers as a single string
                image_url = image_path_to_url(ds_2_img[train_ds_name], sample['image'], folder_id)  # Convert image path to URL

                cells.append(Cell(row=i, col=1, value=train_ds_split))
                cells.append(Cell(row=i, col=2, value=sample['question_id']))  # Column A: question_id
                cells.append(Cell(row=i, col=3, value=question))             # Column B: question
                cells.append(Cell(row=i, col=4, value=answers))             # Column C: answers
                cells.append(Cell(row=i, col=5, value=image_url))             # Column D: image URL

            current_sheet.update_cells(cells)

            list_of_lists = current_sheet.get_all_values()
            last_row = len(list_of_lists)
            current_sheet.update_cell(last_row + 1, 5, "Test")
            cells = [] 

            folder_metadata = {
                'name': f"{hidden_layer_name[c]}_intersect_{test_ds_split}",
                'mimeType': 'application/vnd.google-apps.folder'
            }
            folder = service.files().create(
                body=folder_metadata,
                fields='id'
            ).execute()
            folder_id = folder.get('id')

            permission = {
                'type': 'anyone',
                'role': 'reader'
            }
            service.permissions().create(
                fileId=folder_id,
                body=permission
            ).execute()
            print('Folder permissions set to public.')

            folder_link = f'https://drive.google.com/drive/folders/{folder_id}'
            print("Folder_link", folder_link)
            list_of_lists = current_sheet.get_all_values()
            last_row = len(list_of_lists)
            current_sheet.update_cell(last_row + 1, 5, folder_link)
            list_of_lists = current_sheet.get_all_values()
            last_row = len(list_of_lists)

            for i, sample in enumerate(new_test_inter_samples, start=last_row+1):
                question = sample['question']
                answers = ", ".join(sample['answer'])  # Join answers as a single string
                image_url = image_path_to_url(ds_2_img[test_ds_name], sample['image'], folder_id)  # Convert image path to URL

                cells.append(Cell(row=i, col=1, value=test_ds_split))
                cells.append(Cell(row=i, col=2, value=sample['question_id']))  # Column A: question_id
                cells.append(Cell(row=i, col=3, value=question))             # Column B: question
                cells.append(Cell(row=i, col=4, value=answers))             # Column C: answers
                cells.append(Cell(row=i, col=5, value=image_url))             # Column D: image URL

            current_sheet.update_cells(cells)

            list_of_lists = current_sheet.get_all_values()
            last_row = len(list_of_lists)
            current_sheet.update_cell(last_row + 1, 1, "Done ==========")


            range_label = {
                "test_top" : test_top_range, 
                "intersect" : inter_range, 
                "left_tail" : test_left_range

            } 
            colors = ["green", "red", "yellow"] 
            print("test top range", test_top_range)
            print("intersect range", inter_range)
            print("left tail", test_left_range)
          
            for i, (label_name, (start, end)) in enumerate(range_label.items()): 
                plt.axvline(x=start, linestyle='--', color=colors[i], alpha=0.7)
                plt.axvline(x=end,linestyle='--', color=colors[i], alpha=0.7)


            plt.text(x=0.95, y=0.95, s='Green = Top\nRed = Intersect\nYellow = Tail',
            horizontalalignment='right', verticalalignment='top',
            transform=plt.gca().transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

            plt.title(f'Histogram {hidden_layer_name[c]}  - ({train_ds_split}, {test_ds_split})')
            
            plt.xlabel('Maha score')
            plt.ylabel('Frequency')
            plt.legend()
            plt.savefig(f"/nethome/bmaneech3/flash/vlm_robustness/result_output/contextual_ood/hist_plot/annot_{hidden_layer_name[c]}_{train_ds_split}_{test_ds_split}.jpg")
            plt.close()

            time.sleep(30)

            #add to google sheets



#steps to retrieve samples from different parts 

"""
1) 
2) get top region for each -> select samples from that - 50
    - mark [left,right bound to annotate histogram] -> store left right bound 
3) get intersect region - 50 

4) get outlier regions - 50 -> left tail 





tensor : size (concept, value) -> get index 
"""












