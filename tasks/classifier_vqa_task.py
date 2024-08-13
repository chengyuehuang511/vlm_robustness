import json
import logging
import numpy as np
import os

import lavis.common.dist_utils as dist_utils
from data.classifier_vqa_dataset import ClassifierVQADataset
from tasks.vqa_task_utils import save_vqa_output, after_predict_answers_valid_step, QAOutput
from lavis.common.registry import registry
from lavis.tasks.base_task import BaseTask
from packg.iotools.jsonext import load_json
from metrics.load_metrics import setup_clipmatch_metrics, setup_text_metrics
from metrics.torchmetrics_ext import MetricCollectionExt
from metrics.preprocessing import preprocess_text_simple


def get_generation_kwargs(run_cfg):
    return dict(
        num_beams=run_cfg["num_beams"],  # default 3
        max_new_tokens=run_cfg["max_len"],  # old was max_len=10
        min_new_tokens=run_cfg["min_len"],  # old was min_len=1
        prompt=run_cfg["prompt"],  # default ""
        length_penalty=run_cfg.get("length_penalty", -1),
        use_nucleus_sampling=run_cfg.get("use_nucleus_sampling", False),
        temperature=run_cfg.get("temperature", 1.0),
        top_p=run_cfg.get("top_p", 0.9),
        inference_method=run_cfg.get("inference_method", "generate"),
        num_ans_candidates=run_cfg.get("num_ans_candidates", 128),  # used for "rank" method
        repetition_penalty=run_cfg.get("repetition_penalty", 1.0),
    )
    

@registry.register_task("classifier_vqa")
class ClassifierVQATask(BaseTask):
    def __init__(self, cfg):
        super().__init__()
        run_cfg = cfg.run_cfg
        self.evaluate = run_cfg["evaluate"]  # default False
        self.generation_kwargs = get_generation_kwargs(run_cfg)
        self.answer_list = None
        self.ques_files = dict()
        self.anno_files = dict()

    @classmethod
    def setup_task(cls, cfg):
        return cls(cfg)

    def build_datasets(self, cfg):
        datasets = super().build_datasets(cfg)
        self.annotation, self.answer_list = get_anno_for_classifier_vqa(datasets)
        if len(self.ques_files) > 0:
            assert len(self.ques_files) == len(
                self.anno_files
            ), "Only support one split for evaluation."
        return datasets

    def valid_step(self, model, samples):
        qa_output: QAOutput = model.predict_answers(
            samples=samples,
            return_dict=True,
            answer_list=self.answer_list,
            **self.generation_kwargs,
        )
        # print("qa_output", qa_output)
        return after_predict_answers_valid_step(samples, qa_output)

    def after_evaluation(self, val_result, split_name, **kwargs):
        result_file = save_vqa_output(self, val_result, split_name)
        metrics = self._report_metrics(result_file=result_file, split=split_name)
        return metrics

    @dist_utils.main_process
    def _report_metrics(self, result_file, split):
        """
        Calculate accuracy
        """
        anno = self.annotation[split]
        anno_dict = convert_list_to_dict(anno, "class_idx")  # dict {question_id: class_idx}

        results = load_json(result_file)  # list of {"question_id": int, "answer": str}
        # results rearrange according to question_id's ascending order
        results = sorted(results, key=lambda x: int(x["question_id"]))
        results_dict = convert_list_to_dict(results, "answer")  # dict {question_id: answer}

        labels = self.answer_list  # ['earth star fungus', 'hen of the woods mushroom', 'bolete', 'corn cob', 'toilet paper']
        metrics = eval_classifier_vqa(results_dict, anno_dict, labels)

        with open(os.path.join(registry.get_path("output_dir"), "evaluate.txt"), "a") as f:
            f.write(json.dumps(metrics) + "\n")

        return metrics


def all_metrics(results_dict, anno_dict, labels):
    """
    Args:
        results_dict: dict {question_id: answer}
        anno_dict: dict {question_id: class_idx}
        labels: list of label str

    Returns:
        metrics dict
    """
    target_ids = [class_idx for qid, class_idx in anno_dict.items()]
    target_text = [labels[class_idx] for class_idx in target_ids]
    values = [answer for qid, answer in results_dict.items()]
    keys = list(results_dict.keys())

    # load metrics
    clipmatch_metrics: MetricCollectionExt = setup_clipmatch_metrics(labels)
    text_metrics: MetricCollectionExt = setup_text_metrics()

    results = {}
    # compute clipmatch metrics
    for metric_name, metric in clipmatch_metrics.items():
        metric.reset()
        metric.update(keys, values, target_ids)
        score = metric.compute()
        results[metric_name] = score
        logging.info(f"{metric_name}: {score}")
    
    # compute text metrics
    for metric_name, metric in text_metrics.items():
        metric.reset()
        metric.update(values, target_text)
        score = metric.compute()
        results[metric_name] = score
        logging.info(f"{metric_name}: {score}")
    
    return results


def get_anno_for_classifier_vqa(datasets):
    annotation = {}
    answer_list = None
    for dataset in datasets.values():
        for split in dataset:
            dset: ClassifierVQADataset = dataset[split]
            annotation[split] = dset.annotation
            answer_list = dataset[split].answer_list
    assert answer_list is not None, "Answer list is not available."
    return annotation, answer_list


def convert_list_to_dict(list_data, value_field, key_field="question_id"):
    dict_data = {str(item[key_field]): item[value_field] for item in list_data}
    return dict_data


def eval_classifier_vqa(results_dict, anno_dict, labels):
    """

    Args:
        results_dict: dict {question_id: answer}
        anno_dict: dict {question_id: class_idx}
        labels: list of label str

    Returns:
        metrics dict
    """

    for key, val in results_dict.items():
        results_dict[key] = preprocess_text_simple(val)

    labels = [preprocess_text_simple(label) for label in labels]

    # for the answer score style eval the labels are the answer list
    # and each answer is one of the label

    metrics = {}
    label2idx = {label: idx for idx, label in enumerate(labels)}

    # print("label2idx", label2idx)
    # classes_data = load_json("/nethome/chuang475/flash/projects/vlm_robustness/tmp/datasets/imagenet/ovqa/ovqa/annotations/imagenet1k/generated/classes_data.json")
    # new_label2idx = {i["clip_bench_label"]: i["class_idx"] for i in classes_data}
    # print("new_label2idx", new_label2idx)
    # assert label2idx == new_label2idx

    # naive eval: either hit the label perfectly or the answer is wrong.
    # if the answer is not in the label list, it is considered as wrong. (label idx -1)
    pred_labels = np.array([label2idx.get(answer, -1) for qid, answer in results_dict.items()])

    # HACK to correct int and str missmatch type
    # print("anno_dict", list(anno_dict.keys())[0], type(list(anno_dict.keys())[0]))
    # print("results_dict", list(results_dict.keys())[0], type(list(results_dict.keys())[0]))
    if type(list(anno_dict.keys())[0]) != type(list(results_dict)[0]):
        # convert everything to str
        if isinstance(list(anno_dict.keys())[0], int):
            new_anno_dict = {str(key): val for key, val in anno_dict.items()}
            anno_dict = new_anno_dict
            del new_anno_dict
        if isinstance(list(results_dict.keys())[0], int):
            new_results_dict = {str(key): val for key, val in results_dict.items()}
            results_dict = new_results_dict
            del new_results_dict

    assert list(anno_dict.keys()) == list(results_dict.keys()), (
        f"Mismatched keys between predictions and annotations."
        f"\n========== annotations:\n{anno_dict}\n\n========== predictions:\n{results_dict}"
    )
    gt_labels = np.array([label for qid, label in anno_dict.items()])  # TODO
    acc = np.mean(gt_labels == pred_labels)
    metrics["acc"] = acc

    for i, (k, class_idx) in enumerate(anno_dict.items()):
        if i >= 5:
            break
        result = results_dict[k]
        label = labels[class_idx]
        logging.info(f"Example {i} {k}: pred {result} gt {label}")

    # print len of answers
    logging.info(f"Answer count: {len(results_dict)}")  # 50000

    n_invalid = np.sum(pred_labels == -1)  # can't find exact match in label list
    n_total = len(pred_labels)

    logging.info(f"Invalid answer count: {n_invalid}/{n_total} ({n_invalid / n_total:.4%})")
    logging.info("Classification accuracy is: %.04f" % acc)

    # valid answer classification accuracy
    valid_mask = pred_labels != -1
    valid_acc = np.mean(gt_labels[valid_mask] == pred_labels[valid_mask])
    metrics["valid_acc"] = valid_acc
    logging.info(f"Valid answer classification accuracy is: %.04f\n" % valid_acc)
    
    metrics.update(all_metrics(results_dict, anno_dict, labels))

    metrics["agg_metrics"] = metrics["ClipM1"]  # model selection metric
    
    return metrics

# if __name__ == "__main__":
#     results_file = load_json("/coc/pskynet4/chuang475/projects/LAVIS/lavis/output/PALIGEMMA/IMAGENET1K/20240810190/result/val_vqa_result.json")
#     anno = load_json("/nethome/chuang475/flash/projects/vlm_robustness/tmp/datasets/imagenet/ovqa/ovqa/annotations/imagenet1k/generated/val.json")
#     anno_dict = convert_list_to_dict(anno, "class_idx")

#     results = load_json(results_file)  # list of {"question_id": int, "answer": str}
#     results_dict = convert_list_to_dict(results, "answer")

#     labels = load_json("/nethome/chuang475/flash/projects/vlm_robustness/tmp/datasets/imagenet/ovqa/ovqa/annotations/imagenet1k/generated/classes_data.json")
#     metrics = eval_classifier_vqa(results_dict, anno_dict, labels)