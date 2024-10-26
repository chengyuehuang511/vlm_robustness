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
spreadsheet = client.open_by_key('10EpcWYo6Qng-eQ3dxz4SILZNRWlPGCrCIww45k-3yEg')


SERVICE_ACCOUNT_FILE = '/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/credential.json'
SCOPES = ['https://www.googleapis.com/auth/drive.file']
credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES)
service = build('drive', 'v3', credentials=credentials)

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
    incorrect_samples = vqa_scorer.evaluate() #list of question_ids

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
    return metrics, incorrect_samples 

dataset_to_res_path = {
    'vqa_v2_val': '/coc/pskynet4/chuang475/projects/LAVIS/lavis/output/PALIGEMMA/VQA/20240710023/result/val_vqa_result.json',
    'ivvqa_test': '/coc/pskynet4/chuang475/projects/LAVIS/lavis/output/PALIGEMMA/IV-VQA/20240716004/result/val_vqa_result.json',
    'cvvqa_test': '/coc/pskynet4/chuang475/projects/LAVIS/lavis/output/PALIGEMMA/CV-VQA/20240716003/result/val_vqa_result.json',
    'vqa_ce_test': '/coc/pskynet4/chuang475/projects/LAVIS/lavis/output/PALIGEMMA/VQA_CE/20240716004/result/val_vqa_result.json',
    'advqa_test': '/coc/pskynet4/chuang475/projects/LAVIS/lavis/output/PALIGEMMA/ADVQA/20240717192/result/val_vqa_result.json',
    'textvqa_test': '/coc/pskynet4/chuang475/projects/LAVIS/lavis/output/PALIGEMMA/TextVQA/20240717232/result/val_vqa_result.json',
    'okvqa_test': '/coc/pskynet4/chuang475/projects/LAVIS/lavis/output/PALIGEMMA/OKVQA/20240718013/result/val_vqa_result.json',
    'vqa_cp_test': '/coc/pskynet4/chuang475/projects/LAVIS/lavis/output/PALIGEMMA/VQACP/20240716022/result/val_vqa_result.json', 
    'vizwiz_test'  : '', 
}
dataset_answer_path = {
    'vqa_v2_val': '/coc/pskynet6/chuang475/.cache/lavis/coco/annotations/v2_mscoco_val2014_annotations.json',
    'ivvqa_test': '/coc/pskynet4/chuang475/projects/vlm_robustness/tmp/datasets/iv-vqa/val/BS/vedika2/nobackup/thesis/mini_datasets_qa/0.1_0.1/v2_mscoco_val2014_annotations.json',
    'cvvqa_test': '/coc/pskynet4/chuang475/projects/vlm_robustness/tmp/datasets/cv-vqa/val/BS/vedika2/nobackup/thesis/mini_datasets_qa_COUNTING_DEL_1/0.1_0.0/v2_mscoco_val2014_annotations.json',
    'vqa_ce_test': '/coc/pskynet4/chuang475/projects/vlm_robustness/tmp/datasets/vqace/val/annotation_subset.json',
    'advqa_test': '/coc/pskynet4/chuang475/projects/vlm_robustness/tmp/datasets/advqa/val/v1_mscoco_val2017_advqa_annotations_new.json',
    'textvqa_test': '/coc/pskynet4/chuang475/projects/vlm_robustness/tmp/datasets/textvqa/val/annotation.json',
    'okvqa_test': '/srv/datasets/ok-vqa_dataset/mscoco_val2014_annotations.json',
    'vqa_cp_test': '/coc/pskynet4/chuang475/projects/vlm_robustness/tmp/datasets/vqacp2/test/annotation_new.json',
    'vizwiz_test' : '/coc/pskynet4/chuang475/projects/vlm_robustness/tmp/datasets/vizwiz/val/annotation.json'

}

dataset_question_path = { 
    'vqa_v2_val': '/coc/pskynet6/chuang475/.cache/lavis/coco/annotations/v2_OpenEnded_mscoco_val2014_questions.json',
    'ivvqa_test': '/coc/pskynet4/chuang475/projects/vlm_robustness/tmp/datasets/iv-vqa/val/BS/vedika2/nobackup/thesis/mini_datasets_qa/0.1_0.1/v2_OpenEnded_mscoco_val2014_questions.json',
    'cvvqa_test': '/coc/pskynet4/chuang475/projects/vlm_robustness/tmp/datasets/cv-vqa/val/BS/vedika2/nobackup/thesis/mini_datasets_qa_COUNTING_DEL_1/0.1_0.0/v2_OpenEnded_mscoco_val2014_questions.json',
    'vqa_ce_test': '/coc/pskynet4/chuang475/projects/vlm_robustness/tmp/datasets/vqace/val/question_subset.json',
    'advqa_test': '/coc/pskynet4/chuang475/projects/vlm_robustness/tmp/datasets/advqa/val/v1_OpenEnded_mscoco_val2017_advqa_questions.json',
    'textvqa_test': '/coc/pskynet4/chuang475/projects/vlm_robustness/tmp/datasets/textvqa/val/question.json',
    'okvqa_test': '/srv/datasets/ok-vqa_dataset/OpenEnded_mscoco_val2014_questions.json' , 
    'vqa_cp_test' : '/coc/pskynet4/chuang475/projects/vlm_robustness/tmp/datasets/vqacp2/test/question_new.json',
    'vizwiz_test' : '/coc/pskynet4/chuang475/projects/vlm_robustness/tmp/datasets/vizwiz/val/question.json'
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

left_region = {
    'vqa_v2_val': (-80.0, -51.095001220703125), 
    'ivvqa_test': (-73.0, -48.15999984741211), 
    'cvvqa_test': (-62.75, -46.07749938964844), 
    'vqa_ce_test': (-79.0, -54.70750045776367),
    'advqa_test': (-72.5, -54.310001373291016),
    'textvqa_test': (-112.5, -71.61750030517578),
    'okvqa_test': (-70.0, -56.5)

}

peak_region = { 
    'vqa_v2_val': (-29.34000015258789, -26.725000381469727), 
    'ivvqa_test': (-30.260000228881836, -27.719999313354492), 
    'cvvqa_test': (-28.267499923706055, -25.84000015258789), 
    'vqa_ce_test': (-30.822500228881836, -28.229999542236328),
    'advqa_test': (-31.770000457763672, -29.235000610351562),
    'textvqa_test': (-54.842498779296875, -51.95375061035156),
    'okvqa_test': (-33.0, -30.5)
}

right_region = { 
    'vqa_v2_val': (-21.575000762939453, 0), 
    'ivvqa_test': (-21.700000762939453,0 ), 
    'cvvqa_test': (-21.709999084472656,0), 
    'vqa_ce_test': (-22.712499618530273,0) ,
    'advqa_test': (-23.280000686645508,0),
    'textvqa_test': (-28.957500457763672,0),
    'okvqa_test': (-23.5,0)
}
concept = "image"

tensor_path = "/nethome/bmaneech3/flash/vlm_robustness/result_output/contextual_ood/indiv_result/advqa_test_joint_image.pth"


#want to see where most incorrect samples are from
"""
1) get incorrect samples -> list of ques_id 
2) ques_id to maha score 
    - iterate through indiv results -> quesId : indiv_result[id]
    - save quesId2Score.json
3) visualize histogram -> number 
"""

def process_qid_2_score(emb_tensor, split, concept) : 
    #create quesId to indiv result  
    #quesId : maha score 

    maha_list = list(emb_tensor)
    qid_2_score = {} 

    with open(ds_split_2_file[split], 'r') as f : 
        data = json.load(f)
        assert(len(maha_list) == len(data), "indiv result and combined json length not equal")

        for i in range(len(maha_list)) : 
            qid_2_score[data[i]["question_id"]] = maha_list[i].item()


    store_path = f"/nethome/bmaneech3/flash/vlm_robustness/result_output/contextual_ood/region_samples/pt_{split}_{concept}_qid_score.json"
    with open(store_path, 'w') as file : 
        json.dump(qid_2_score, file)




def get_ques_id(emb_tensor, thres, split) : 
    st,en = thres
    mask = (emb_tensor >= st) & (emb_tensor <= en) 

    indices = list(torch.nonzero(mask, as_tuple=False).squeeze()) #getting indices 
    # print("INDICESSS", indices)
    ques_id = []

    res_path = dataset_to_res_path[split]

    with open(ds_split_2_file[split], 'r') as f : 
        data = json.load(f)
    
    ques_id = {data[i]["question_id"] : 0 for i in tqdm((indices), desc="Processing Question IDs")}
        # ques_id = {data[i]["question_id"] for i in tqdm(range(len(data)), desc="Processing Question IDs") if i in indices}

    res_samples = []
    with open(res_path, 'r') as f :
        data = json.load(f)

        for sample in tqdm(data, desc="Adding res filtered samples") : 
            # print(sample)
            if sample['question_id'] in ques_id : 
                res_samples.append(sample)

    
    ques_samples = []
    ques_path = dataset_question_path[split]
    with open(ques_path, 'r') as f : 
        data = json.load(f)
        if isinstance(data, dict) : 
            questions = data["questions"]

        for sample in tqdm(questions, desc="Adding ques filtered samples") : 
            # print(sample)
            if sample['question_id'] in ques_id : 
                ques_samples.append(sample)

        if isinstance(data, dict) : 
            data["questions"] = ques_samples 
            ques_samples = data

    ans_path = dataset_answer_path[split]
    ans_samples = []
    with open(ans_path, 'r') as f : 
        data = json.load(f)
        if isinstance(data, dict) : 
            ann = data["annotations"]

        for sample in tqdm(ann, desc="Adding ans filtered samples") : 
            # print(sample)
            if sample['question_id'] in ques_id : 
                ans_samples.append(sample)

        if isinstance(data, dict) : 
            data["annotations"] = ans_samples 
            ans_samples = data

    return indices, res_samples, ques_samples, ans_samples


for dataset, res_path in  dataset_to_res_path.items(): 

    incorrect_dict = {
    } 

    print(f'Running for {dataset}')

    emb_tensor = torch.load(f"/coc/pskynet4/bmaneech3/vlm_robustness/result_output/contextual_ood/pt_indiv_result/{dataset}_joint_image.pth")

    emb_tensor = emb_tensor[0] #get only 1 concept 
    process_qid_2_score(emb_tensor, dataset, concept)
    metric_path = f"/nethome/bmaneech3/flash/vlm_robustness/result_output/contextual_ood/region_samples/result/pt_{dataset}_overall.txt"
    # output_dir = f"/nethome/bmaneech3/flash/vlm_robustness/result_output/contextual_ood/region_samples/result/{dataset}_output.txt"
    metrics, incorrect_samples = report_metrics(dataset_to_res_path[dataset],dataset_question_path[dataset], dataset_answer_path[dataset], metric_path)
    print(metrics)

    #store incorrect samples 
    incorrect_sample_path = f"/nethome/bmaneech3/flash/vlm_robustness/result_output/contextual_ood/region_samples/result/pt_{dataset}_{concept}_incorrect_sample.json"

    with open(incorrect_sample_path, 'w') as file : 
        json.dump(incorrect_samples, file)

    store_path = f"/nethome/bmaneech3/flash/vlm_robustness/result_output/contextual_ood/region_samples/pt_{dataset}_{concept}_qid_score.json"
   
    co = 0 
    score_list = [] 
    with open(store_path, 'r') as file : 
        data= json.load(file)
        incorrect_samples = set(incorrect_samples)
        for quesId in incorrect_samples : 
            if str(quesId) not in data : 
                co+= 1 
                print("not found")
                continue 
            else : 
                score_list.append(data[str(quesId)])

    print(f"Ques Id not found: {co} samples")
    plt.hist(score_list, bins=50)
    plt.xlabel('Mahascore')
    plt.ylabel('Frequency')
    plt.title(f'Histogram {dataset} incorrect sample distribution : {concept} shift')
    plt.savefig(f"/nethome/bmaneech3/flash/vlm_robustness/result_output/contextual_ood/region_samples/result/pt_{dataset}_{concept}_fail_distrib.png")
    plt.clf()


    incorrect_maha_scores = score_list
    num_bins = 100
    """plot incorrect samples as percentage of total samples"""
    all_maha_scores = [elem.item() for elem in emb_tensor]
    all_hist, bin_edges = np.histogram(all_maha_scores, bins=num_bins, range=(min(all_maha_scores), max(all_maha_scores)))
    incorrect_hist, _ = np.histogram(incorrect_maha_scores, bins=num_bins, range=(min(all_maha_scores), max(all_maha_scores)))
    percent_incorrect = np.divide(incorrect_hist, all_hist, where=(all_hist != 0)) * 100
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # Get the center of each bin


    plt.bar(bin_centers, percent_incorrect, width=bin_edges[1] - bin_edges[0], color='orange', alpha=0.7)
    plt.xlabel('Mahalanobis Score')
    plt.ylabel(f'Percentage of Incorrect Samples (%)')
    plt.title(f'Percentage of Incorrect Samples by Maha score bins: {dataset} : {concept} shift')
    plt.savefig(f"/nethome/bmaneech3/flash/vlm_robustness/result_output/contextual_ood/region_samples/result/pt_{dataset}_{concept}_incorrect_perc.png")
    plt.close()




