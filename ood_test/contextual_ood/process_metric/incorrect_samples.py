import logging
from lavis.common.registry import registry
from lavis.common.vqa_tools.vqa import VQA
from lavis.common.vqa_tools.vqa_eval import VQAEval
import random 

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
import numpy as np 
from tqdm import tqdm 
from concurrent.futures import ProcessPoolExecutor


scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]

creds = ServiceAccountCredentials.from_json_keyfile_name('/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/credential.json', scope)

client = gspread.authorize(creds)

#update keys 
spreadsheet = client.open_by_key('10EpcWYo6Qng-eQ3dxz4SILZNRWlPGCrCIww45k-3yEg')


SERVICE_ACCOUNT_FILE = '/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/credential.json'
SCOPES = ['https://www.googleapis.com/auth/drive.file']
credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES)
service = build('drive', 'v3', credentials=credentials)


#res_file_path : ft_method -> test_split -> path
res_file_dict = json.load(open("/nethome/bmaneech3/flash/vlm_robustness/ood_test/other/res_file_path.json"))


def report_metrics(result_file, ques_file, anno_file, file_path):
    """
    Use official VQA evaluation script to report metrics.
    """
    metrics = {}

    vqa = VQA(anno_file, ques_file)
    vqa_result = vqa.loadRes(
        resFile=result_file, quesFile=ques_file
    )
    # create vqaEval object by taking vqa and vqaRes
    # n is precision of accuracy (number of places after decimal), default is 2
    vqa_scorer = VQAEval(vqa, vqa_result, n=2)
    logging.info("Start VQA evaluation.")
    incorrect_samples, dataset = vqa_scorer.evaluate() #list of instance_IDs

    # print accuracies
    overall_acc = vqa_scorer.accuracy["overall"]
    metrics["agg_metrics"] = overall_acc

    logging.info("Overall Accuracy is: %.09f\n" % overall_acc)
    print("Overall Accuracy is: %.09f\n" % overall_acc)
    logging.info("Per Answer Type Accuracy is the following:")

    for ans_type in vqa_scorer.accuracy["perAnswerType"]:
        logging.info(
            "%s : %.09f"
            % (ans_type, vqa_scorer.accuracy["perAnswerType"][ans_type])
        )
        metrics[ans_type] = vqa_scorer.accuracy["perAnswerType"][ans_type]
        print(f"yes/no {metrics[ans_type]:.9f}")


    with open(
        os.path.join(file_path), "a"
    ) as f:
        f.write(json.dumps(metrics) + "\n")

    print(incorrect_samples)
    return metrics, incorrect_samples, dataset



name_map = json.load(open("/nethome/bmaneech3/flash/vlm_robustness/ood_test/other/name_map.json"))

dataset_question_path = { 
    'vqa_v2_val': '/coc/pskynet6/chuang475/.cache/lavis/coco/annotations/v2_OpenEnded_mscoco_val2014_questions.json',
    'ivvqa': '/coc/pskynet4/chuang475/projects/vlm_robustness/tmp/datasets/iv-vqa/val/BS/vedika2/nobackup/thesis/mini_datasets_qa/0.1_0.1/v2_OpenEnded_mscoco_val2014_questions.json',
    'cvvqa': '/coc/pskynet4/chuang475/projects/vlm_robustness/tmp/datasets/cv-vqa/val/BS/vedika2/nobackup/thesis/mini_datasets_qa_COUNTING_DEL_1/0.1_0.0/v2_OpenEnded_mscoco_val2014_questions.json',
    'vqa_ce': '/coc/pskynet4/chuang475/projects/vlm_robustness/tmp/datasets/vqace/val/question_subset.json',
    'advqa': '/coc/pskynet4/chuang475/projects/vlm_robustness/tmp/datasets/advqa/val/v1_OpenEnded_mscoco_val2017_advqa_questions.json',
    'textvqa': '/coc/pskynet4/chuang475/projects/vlm_robustness/tmp/datasets/textvqa/val/question.json',
    'okvqa': '/srv/datasets/ok-vqa_dataset/OpenEnded_mscoco_val2014_questions.json' , 
    'vqa_cp' : '/coc/pskynet4/chuang475/projects/vlm_robustness/tmp/datasets/vqacp2/test/question_new.json',
    'vizwiz' : '/coc/pskynet4/chuang475/projects/vlm_robustness/tmp/datasets/vizwiz/val/question.json',
    'vqa_rephrasings' : '/srv/datasets/vqa_rephrasings/v2_OpenEnded_mscoco_valrep2014_humans_og_questions.json'
}

dataset_answer_path = {
    'vqa_v2_val': '/coc/pskynet6/chuang475/.cache/lavis/coco/annotations/v2_mscoco_val2014_annotations.json',
    'ivvqa': '/coc/pskynet4/chuang475/projects/vlm_robustness/tmp/datasets/iv-vqa/val/BS/vedika2/nobackup/thesis/mini_datasets_qa/0.1_0.1/v2_mscoco_val2014_annotations.json',
    'cvvqa': '/coc/pskynet4/chuang475/projects/vlm_robustness/tmp/datasets/cv-vqa/val/BS/vedika2/nobackup/thesis/mini_datasets_qa_COUNTING_DEL_1/0.1_0.0/v2_mscoco_val2014_annotations.json',
    'vqa_ce': '/coc/pskynet4/chuang475/projects/vlm_robustness/tmp/datasets/vqace/val/annotation_subset.json',
    'advqa': '/coc/pskynet4/chuang475/projects/vlm_robustness/tmp/datasets/advqa/val/v1_mscoco_val2017_advqa_annotations_new.json',
    'textvqa': '/coc/pskynet4/chuang475/projects/vlm_robustness/tmp/datasets/textvqa/val/annotation.json',
    'okvqa': '/srv/datasets/ok-vqa_dataset/mscoco_val2014_annotations.json',
    'vqa_cp': '/coc/pskynet4/chuang475/projects/vlm_robustness/tmp/datasets/vqacp2/test/annotation_new.json',
    'vizwiz' : '/coc/pskynet4/chuang475/projects/vlm_robustness/tmp/datasets/vizwiz/val/annotation.json',
    'vqa_rephrasings' : '/srv/datasets/vqa_rephrasings/v2_mscoco_valrep2014_humans_og_annotations.json'
}

ft_methods = ["fft", "lora", "lp", "lpft", "ftp", "spd", "digrap", "pt_emb"] 
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

#incorrect percentage & heatmap visualization 

"""for each ft 
    - results_dict : test_split -> concept -> score (int)
        for each split 
        - indiv_results : concept -> instance_id -> score (int)
"""

"""
ft_method/

for a ft_method & test split -> get the list of incorrect samples - list of instance ids 

then for each concept -> get the indiv_results concept & then arrange them according to 


get indices from right tail & left tail region -> 
see how much it overlaps 

plot PCA 
"""

ds_2_ann = json.load(open("/nethome/bmaneech3/flash/vlm_robustness/ood_test/other/ds_split_2_file.json"))
incorrect_dict = {}
for  ft_method in ft_methods : 
    incorrect_dir = "/nethome/bmaneech3/flash/vlm_robustness/result_output/contextual_ood/incorrect_dict.json"

    if os.path.exists(incorrect_dir) :
        with open(incorrect_dir, 'r') as file : 
            incorrect_dict = json.load(file)

    else : 
        incorrect_dict = {} 

    incorrect_dict[ft_method] = {} 
    
    for test_split in test_splits : 

        print(f"Running for {test_split}")
        if ft_method not in res_file_dict or test_split not in res_file_dict[ft_method] : 
            continue 
        
        #get incorrect samples 
        #TODO verify that instance ids correspond to the combined data
        try : 
            metrics, incorrect_samples, dataset = report_metrics(res_file_dict[ft_method][test_split], dataset_question_path[name_map[test_split]], dataset_answer_path[name_map[test_split]], "/nethome/bmaneech3/flash/vlm_robustness/ood_test/other/overall.txt")

        except Exception as e :
            print(f"Issues with path for {ft_method} {test_split}")
            print(e)
            continue 

        dataset_ann = dataset["annotations"]
        correct_ann = json.load(open(ds_2_ann[name_map[test_split]]))

        for ann1, ann2 in zip(dataset_ann, correct_ann) : 
            if ann1["question_id"] != ann2["question_id"] : 
                raise Exception("question id mismatch")

        #incorrect_samples = list of Instance IDs
        incorrect_dict[ft_method][test_split] = incorrect_samples 

        #perform plots 
        for concept in concept_list : 
            indiv_res_path = f"/nethome/bmaneech3/flash/vlm_robustness/result_output/contextual_ood/new_ft_indiv_result/{ft_method}/{test_split}_indiv_result.pth"
            indiv_res_dict = torch.load(indiv_res_path)

            # score_list = [ value for key, value in indiv_res_dict[concept].items() if key in ]

            #verify uniqueness
            vis = {}
            score_list = [] 
            for instance_id in incorrect_samples : 
                if instance_id in vis : 
                    raise Exception("Found duplicate instance id")
                
                score_list.append(indiv_res_dict[concept][instance_id].to(dtype=torch.float32).item())
                vis[instance_id] = True 
            #plot incorrect samples as percentage of total samples 

            num_bins = 50 
            all_maha_scores = [value.to(dtype=torch.float32).item() for _, value in indiv_res_dict[concept].items()]

            all_hist, bin_edges = np.histogram(all_maha_scores, bins=num_bins, range=(min(all_maha_scores), max(all_maha_scores)))

            incorrect_hist, _ = np.histogram(score_list, bins=num_bins, range=(min(all_maha_scores), max(all_maha_scores)))
            percent_incorrect = np.divide(incorrect_hist, all_hist, where=(all_hist != 0)) * 100
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # Get the center of each bin
            plt.bar(bin_centers, percent_incorrect, width=bin_edges[1] - bin_edges[0], color='orange', alpha=0.7)
            plt.xlabel('Mahalanobis Score')
            plt.ylabel(f'Percentage of Incorrect Samples (%)')
            plt.title(f'Percentage of Incorrect Samples across Histogram')

            hist_dir = f"/nethome/bmaneech3/flash/vlm_robustness/result_output/contextual_ood/region_samples/new_incorrect_diag/{ft_method}/{concept}"
            os.makedirs(hist_dir, exist_ok=True)

            hist_path = os.path.join(hist_dir, f"{test_split}_incorrect_perc.png")
    
            plt.savefig(hist_path)
            plt.close()
            print("plotted")

    with open(incorrect_dir, 'w') as file : 
        json.dump(incorrect_dict, file)











