import logging
import json
import os
import torch
from tqdm import tqdm

from lavis.common.registry import registry
from lavis.common.vqa_tools.vqa import VQA
from lavis.common.vqa_tools.vqa_eval import VQAEval

def report_metrics(result_file, ques_file, anno_file):
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
    vqa_scorer.evaluate()

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
        os.path.join("/nethome/bmaneech3/flash/vlm_robustness/result_output", "vqa_vs_ft.txt"), "a"
    ) as f:
        f.write(json.dumps(metrics) + "\n")
    return metrics

if __name__ == "__main__":
    # report_metrics(
    #     "/coc/pskynet4/chuang475/projects/LAVIS/lavis/output/ALBEF/VQACP/20240606120/result/val_vqa_result.json",
    #     "/nethome/chuang475/flash/projects/vlm_robustness/tmp/datasets/vqacp2/test/question_new.json",
    #     "/nethome/chuang475/flash/projects/vlm_robustness/tmp/datasets/vqacp2/test/annotation_new.json",
    # )
    # file_list = [
    #     "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/KO/test_questions.json", 
    # "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/KOP/test_questions.json",
    # "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/KW/test_questions.json", 
    # "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/KW+KO/test_questions.json", 
    # "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/KWP/test_questions.json", 
    # "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/QT/test_questions.json", 
    # "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/QT+KO/test_questions.json",
    # "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/QT+KW/test_questions.json",
    # "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/OOD-Test/QT+KW+KO/test_questions.json"
    # ] 

    # for file in file_list :
        
        # report_metrics( 
        #     os.path.join(os.path.dirname(file), "test_result.json"),
        #     os.path.join(os.path.dirname(file), "test_questions.json"), 
        #     os.path.join(os.path.dirname(file), "test_annotations.json")
        # )
    

    report_metrics(
        "/nethome/bmaneech3/flash/LAVIS/lavis/output/PALIGEMMA/VQAVS/FT/20240716130/result/test_vqa_result.json",
        "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/test_questions.json",
        "/nethome/bmaneech3/flash/vlm_robustness/tmp/datasets/vqavs/test/test_annotations.json"
    )