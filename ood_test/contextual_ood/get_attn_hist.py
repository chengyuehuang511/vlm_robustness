import logging
from lavis.common.registry import registry
from lavis.common.vqa_tools.vqa import VQA
from lavis.common.vqa_tools.vqa_eval import VQAEval
import random 
import pandas as pd

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

creds = ServiceAccountCredentials.from_json_keyfile_name('/coc/pskynet4/bmaneech3/vlm_robustness/tmp/datasets/credential.json', scope)

client = gspread.authorize(creds)
spreadsheet = client.open_by_key('10EpcWYo6Qng-eQ3dxz4SILZNRWlPGCrCIww45k-3yEg')


SERVICE_ACCOUNT_FILE = '/coc/pskynet4/bmaneech3/vlm_robustness/tmp/datasets/credential.json'
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
    'vqa_rephrasings_test': '',
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
    "textvqa_train" : "/coc/pskynet4/bmaneech3/vlm_robustness/tmp/datasets/textvqa/train/combined_data.json", 
    "vizwiz_test" :  "/coc/pskynet4/chuang475/projects/vlm_robustness/tmp/datasets/vizwiz/val/combined_data.json",
    "vqa_ce_test" : "/coc/pskynet4/chuang475/projects/vlm_robustness/tmp/datasets/vqace/val/combined_data_subset.json",
    "vqa_cp_train" : "/coc/pskynet4/chuang475/projects/vlm_robustness/tmp/datasets/vqacp2/train/vqacp_v2_train_questions.json", 
    "vqa_cp_test" : "/coc/pskynet4/chuang475/projects/vlm_robustness/tmp/datasets/vqacp2/test/combined_data.json", 
    "vqa_lol_train": "/coc/pskynet4/bmaneech3/vlm_robustness/tmp/datasets/vqalol/train/combined_data.json",
    "vqa_lol_test": "/coc/pskynet4/bmaneech3/vlm_robustness/tmp/datasets/vqalol/test/combined_data.json", 
    "vqa_rephrasings_test": "/coc/pskynet4/chuang475/projects/vlm_robustness/tmp/datasets/vqa_rephrasings/combined_data.json",
    "vqa_vs_train" : "/coc/pskynet4/bmaneech3/vlm_robustness/tmp/datasets/vqavs/train/combined_data.json", 
    "vqa_vs_id_val" : "/coc/pskynet4/bmaneech3/vlm_robustness/tmp/datasets/vqavs/val/combined_data.json", 
    "vqa_vs_id_test" : "/coc/pskynet4/bmaneech3/vlm_robustness/tmp/datasets/vqavs/test/combined_data.json", 
    "vqa_vs_ood_test" : "/coc/pskynet4/bmaneech3/vlm_robustness/tmp/datasets/vqavs/test/combined_data.json", 
    "vqa_vs_KO" : "/coc/pskynet4/bmaneech3/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/KO/combined_data.json",
    "vqa_vs_KOP" : "/coc/pskynet4/bmaneech3/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/KOP/combined_data.json", 
    "vqa_vs_KW" : "/coc/pskynet4/bmaneech3/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/KW/combined_data.json", 
    "vqa_vs_KW_KO" : "/coc/pskynet4/bmaneech3/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/KW+KO/combined_data.json", 
    "vqa_vs_KWP" : "/coc/pskynet4/bmaneech3/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/KWP/combined_data.json", 
    "vqa_vs_QT" : "/coc/pskynet4/bmaneech3/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/QT/combined_data.json", 
    "vqa_vs_QT_KO" : "/coc/pskynet4/bmaneech3/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/QT+KO/combined_data.json", 
    "vqa_vs_QT_KW" : "/coc/pskynet4/bmaneech3/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/QT+KW/combined_data.json", 
    "vqa_vs_QT_KW_KO" : "/coc/pskynet4/bmaneech3/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/QT+KW+KO/combined_data.json"
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
concept = "joint"

tensor_path = "/coc/pskynet4/bmaneech3/vlm_robustness/result_output/contextual_ood/indiv_result/advqa_test_joint_image.pth"


#want to see where most incorrect samples are from
"""
1) get incorrect samples -> list of ques_id 
2) ques_id to maha score 
    - iterate through indiv results -> quesId : indiv_result[id]
    - save quesId2Score.json
3) visualize histogram -> number 
"""

def process_qid_2_score(emb_tensor, split, token): 
    #create quesId to indiv result  
    #quesId : maha score 

    maha_list = list(emb_tensor)
    qid_2_score = {} 

    with open(ds_split_2_file[split], 'r') as f : 
        data = json.load(f)
        assert(len(maha_list) == len(data), "indiv result and combined json length not equal")

        for i in range(len(maha_list)) : 
            qid_2_score[data[i]["question_id"]] = maha_list[i]

    store_path = f"/coc/testnvme/chuang475/projects/vlm_robustness/ood_test/contextual_ood/region_samples/{split}_{token}_attn_score.json"
    if os.path.exists(store_path) :
        print(f"File {store_path} already exists")
    else :
        with open(store_path, 'w') as file : 
            json.dump(qid_2_score, file)

ood_threshold = 60

for gt_dist in ["fft"]: #, "pt_emb"]:
    for method in ["ftp", "lora", "lp", "lpft", "pt_emb", "spd"]:  # "fft", "ftp", "lora", "lp", "lpft", "pt_emb", "spd"
        img_ratio_id_all = []
        txt_ratio_id_all = []

        img_ratio_ood_all = []
        txt_ratio_ood_all = []

        img_ratio_all = []
        txt_ratio_all = []

        for dataset, res_path in  dataset_to_res_path.items(): 

            incorrect_dict = {
            } 

            if dataset.startswith("textvqa") or dataset.startswith("vizwiz"):
                if "test" in dataset:
                    dataset_pth = dataset.replace("test", "val")
            elif dataset.startswith("vqa_v2_val"):
                dataset_pth = "coco_vqa_raw_val"
            elif dataset.startswith("ivvqa"):
                dataset_pth = "coco_iv-vqa_val"
            elif dataset.startswith("cvvqa"):
                dataset_pth = "coco_cv-vqa_val"
            else:
                if "test" in dataset:
                    # substitute test with val
                    dataset_pth = "coco_" + dataset.replace("test", "val")
                else:
                    dataset_pth = "coco_"

            print(f'Running for {dataset}')

            emb_tensor = torch.load(f"/coc/pskynet4/bmaneech3/vlm_robustness/result_output/contextual_ood/new_ft_indiv_result/{gt_dist}/{dataset_pth}_indiv_result.pth")  # gt distance score
            emb_tensor = emb_tensor[concept]

            attn_file = f"/coc/testnvme/chuang475/projects/vlm_robustness/ood_test/contextual_ood/xttn_results/{method}/{dataset_pth}.csv"
            img_ratio = pd.read_csv(attn_file)['img_ratio'].to_list()
            txt_ratio = pd.read_csv(attn_file)['txt_ratio'].to_list()

            print(dataset)
            
            print("pth length", len(emb_tensor))
            print("img_ratio length", len(img_ratio))

            # emb_tensor = emb_tensor[0] #get only 1 concept 
            
            # process_qid_2_score(img_ratio, dataset, "image")
            # process_qid_2_score(txt_ratio, dataset, "text")

            img_ratio_list = np.array(img_ratio)
            txt_ratio_list = np.array(txt_ratio)
            qid_score_list = []

            for i in range(len(emb_tensor)) :
                qid_score_list.append(emb_tensor[i].item())

            qid_score_list = np.array(qid_score_list)

            # get the corresponding img_ratio and txt_ratio for qid_scre > ood_threshold
            img_ratio_ood_all += img_ratio_list[qid_score_list > ood_threshold].tolist()
            txt_ratio_ood_all += txt_ratio_list[qid_score_list > ood_threshold].tolist()

            img_ratio_id_all += img_ratio_list[qid_score_list <= ood_threshold].tolist()
            txt_ratio_id_all += txt_ratio_list[qid_score_list <= ood_threshold].tolist()

            img_ratio_all += img_ratio_list.tolist()
            txt_ratio_all += txt_ratio_list.tolist()

            # """

            # print("intersect length", len(img_ratio_list))

            # Define the number of bins
            num_bins = 50
            # all_maha_scores = [elem.item() for elem in emb_tensor]

            # Compute bin indices based on qid_score_list
            # bins_ = np.linspace(min(all_maha_scores), max(all_maha_scores), num_bins + 1)
            print(f"qid_score_list max: {qid_score_list.max()}, corresponding index {np.argmax(qid_score_list)}")
            print(f"qid_score_list min: {qid_score_list.min()}, corresponding index {np.argmin(qid_score_list)}")
            # bins = np.histogram(qid_score_list, bins=num_bins, range=(min(qid_score_list), max(qid_score_list)+5))[1]
            bins = np.histogram(qid_score_list, bins=num_bins, range=(15, 105))[1]
            # print("bins[-1]", bins[-1])
            bin_indices = np.digitize(qid_score_list, bins) - 1
            print("bin_indices max", bin_indices.max())
            print("bin_indices min", bin_indices.min())

            # Calculate mean img_ratio and txt_ratio for each bin
            img_ratio_means = [img_ratio_list[bin_indices == i].mean() if len(img_ratio_list[bin_indices == i])>0 else 0 for i in range(num_bins)]
            txt_ratio_means = [txt_ratio_list[bin_indices == i].mean() if len(txt_ratio_list[bin_indices == i])>0 else 0 for i in range(num_bins)]

            # Plotting
            # plt.figure(figsize=(10, 6))
            # plt.plot(bins[:-1], img_ratio_means, label='Mean img_ratio', marker='.', markersize=5, linestyle='-', linewidth=1.5)
            # plt.plot(bins[:-1], txt_ratio_means, label='Mean txt_ratio', marker='*', markersize=5, linestyle='-', linewidth=1.5)
            # plt.xlabel("Mahalanobis Score", fontsize=16)
            # plt.ylabel("Mean Ratio", fontsize=16)
            # plt.title("Mean img_ratio and txt_ratio per Mahalanobis Score bin", fontsize=16)
            # plt.legend(fontsize=14)
            # plt.savefig(f"/coc/testnvme/chuang475/projects/vlm_robustness/ood_test/contextual_ood/region_samples/new_attention_results/{dataset}_{concept}_attn_ratio.png")
            # plt.close()

            # Plotting bar plots for img_ratio_means and txt_ratio_means
            plt.bar(bins[:-1], img_ratio_means, width=(bins[1] - bins[0]) * 0.4, label=r'MI$_v$', color='blue', alpha=0.6, align='center')
            plt.bar(bins[:-1] + (bins[1] - bins[0]) * 0.4, txt_ratio_means, width=(bins[1] - bins[0]) * 0.4, label=r'MI$_q$', color='orange', alpha=0.6, align='center')

            # Adding a dotted red line at y=1
            plt.axhline(y=1, color='red', linestyle=':', linewidth=1.5, label='MI=1')

            # y-axis limits
            plt.ylim(0, 4)

            plt.yticks(fontsize=14)
            plt.xticks(fontsize=14)

            # Setting labels, title, and legend
            plt.xlabel(r"Mahalanobis Score (ID $\rightarrow$ OOD)", fontsize=16)
            plt.ylabel("Mean MI", fontsize=16)
            # plt.title("Mean img_ratio and txt_ratio per Mahalanobis Score bin", fontsize=16)
            plt.legend(fontsize=16, loc='upper right')

            # Adding a grid
            plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.7)

            # Shading the background
            plt.gca().set_facecolor('#f5f5f5')  # Light gray background for the plot area

            # Saving the plot
            folder = f"/coc/testnvme/chuang475/projects/vlm_robustness/ood_test/contextual_ood/region_samples/new_legend_attention_results/gt_{gt_dist}/{method}/"
            os.makedirs(folder, exist_ok=True)
            plt.savefig(f"/coc/testnvme/chuang475/projects/vlm_robustness/ood_test/contextual_ood/region_samples/new_legend_attention_results/gt_{gt_dist}/{method}/{dataset}_{concept}_attn_ratio.png")
            plt.close()
            # """
        
        """
        assert len(img_ratio_id_all) == len(txt_ratio_id_all)
        assert len(img_ratio_ood_all) == len(txt_ratio_ood_all)
        assert len(img_ratio_all) == len(txt_ratio_all)
        assert len(img_ratio_id_all) + len(img_ratio_ood_all) == len(img_ratio_all)

        print("method", method)
        print("img_ratio_id", np.array(img_ratio_id_all).mean())
        print("txt_ratio_id", np.array(txt_ratio_id_all).mean())
        print("img_ratio_ood", np.array(img_ratio_ood_all).mean())
        print("txt_ratio_ood", np.array(txt_ratio_ood_all).mean())
        print("img_ratio_all", np.array(img_ratio_all).mean())
        print("txt_ratio_all", np.array(txt_ratio_all).mean())
        """
