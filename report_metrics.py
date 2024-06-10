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

    logging.info("Overall Accuracy is: %.02f\n" % overall_acc)
    logging.info("Per Answer Type Accuracy is the following:")

    for ans_type in vqa_scorer.accuracy["perAnswerType"]:
        logging.info(
            "%s : %.02f"
            % (ans_type, vqa_scorer.accuracy["perAnswerType"][ans_type])
        )
        metrics[ans_type] = vqa_scorer.accuracy["perAnswerType"][ans_type]

    with open(
        os.path.join("/nethome/chuang475/flash/projects/vlm_robustness/result_output", "evaluate.txt"), "a"
    ) as f:
        f.write(json.dumps(metrics) + "\n")
    return metrics

if __name__ == "__main__":
    report_metrics(
        "/coc/pskynet4/chuang475/projects/LAVIS/lavis/output/ALBEF/VQACP/20240606120/result/val_vqa_result.json",
        "/nethome/chuang475/flash/projects/vlm_robustness/tmp/datasets/vqacp2/test/question_new.json",
        "/nethome/chuang475/flash/projects/vlm_robustness/tmp/datasets/vqacp2/test/annotation_new.json",
    )