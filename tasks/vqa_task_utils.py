from collections import defaultdict
import torch
from typing import Any, Dict, List, Tuple, Optional
from lavis.common.registry import registry
from lavis.tasks.base_task import BaseTask
from dataclasses import dataclass
from transformers.utils import ModelOutput


@dataclass
class QAOutput(ModelOutput):
    """
    Args:
        answer: list of predicted answers
        top10_answers_and_probs: either list of top 10 answers and their probabilities
            or list of None
    """

    answer: List[str]
    answers: Optional[List[List[str]]] = None
    labels: Optional[List[List[str]]] = None
    top10_answers_and_probs: List[Optional[Tuple[List[str], List[float]]]] = None


def after_predict_answers_valid_step(samples: Dict[str, Any], qa_output: QAOutput):
    pred_batch = []
    question_ids = samples["question_id"]
    if isinstance(question_ids, torch.Tensor):
        question_ids = question_ids.tolist()

    for batch_index, qid in enumerate(question_ids):
        output_dict = {
            "question_id": qid,
        }
        for output_key, output_value in qa_output.items():
            output_dict[output_key] = output_value[batch_index]
        pred_batch.append(output_dict)
    return pred_batch


def save_vqa_output(
    task: BaseTask,
    val_result: List[Dict[str, Any]],
    split_name,
    id_field_name: str = "question_id",
    vqa_field_name: str = "answer",
    file_identifier: str = "result",
):
    """
    Sort the list of input dictionaries into one list of dicts for each field.
    Save one file for each field.

    Args:
        task:
        val_result: list of dicts like
            [{"question_id": 0, "answers": "string answer", "other...": "other"}, ...]
        split_name:
        id_field_name:
        vqa_field_name:
        file_identifier: result by default or others like followup

    Returns:
        Filename where the vqa answers have been saved

    """

    field_names = [f for f in list(val_result[0].keys()) if f != id_field_name]

    sorted_dicts = defaultdict(list)
    for result_dict in val_result:
        for f in field_names:
            new_f = "vqa" if f == vqa_field_name else f
            sorted_dicts[new_f].append(
                {f: result_dict[f], id_field_name: result_dict[id_field_name]}
            )

    file_names = {}
    for new_f, sorted_dict in sorted_dicts.items():
        _result_file = task.save_result(
            sorted_dict,
            result_dir=registry.get_path("result_dir"),
            filename=f"{split_name}_{new_f}_{file_identifier}",
            remove_duplicate=id_field_name,
            # float_precision=4,
        )
        file_names[new_f] = _result_file

    return file_names["vqa"]