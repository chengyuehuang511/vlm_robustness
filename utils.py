import os
import logging
from typing import Dict, List, Optional
from packg.iotools.jsonext import load_json


def barrier_safe_ext():
    """Barrier only if in a distributed torch run. Does not fail if torch package is missing."""
    if ("RANK" in os.environ or "LOCAL_RANK" in os.environ) and "WORLD_SIZE" in os.environ:
        from torch import distributed as dist

        dist.barrier()


def make_list_smaller(input_list, start_num=0, max_amount=None):
    input_list = input_list[start_num:]
    if max_amount is not None:
        input_list = input_list[:max_amount]
    return input_list


def load_captions_maybe(config, instance_ids=None) -> Optional[Dict[int, List[str]]]:
    # load captions into dataset
    question_caption_file = config.get("question_caption_file", None)
    if question_caption_file is None:
        return None
    # question_caption_file = autoupdate_path(question_caption_file)
    captions = load_json(question_caption_file)
    captions_dict = {caption["question_id"]: caption["caption"] for caption in captions}
    if instance_ids is not None:
        question_captions = {qid: captions_dict[qid] for qid in instance_ids}
        logging.info(f"Loaded captions for {len(instance_ids)} ids from {question_caption_file}")
    else:
        question_captions = captions_dict
        logging.info(f"Loaded all captions, {len(captions_dict)} ids from {question_caption_file}")
    return question_captions


def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    if os.path.exists(log_path) is True:
        os.remove(log_path)
    logger = logging.getLogger()
    
    for handler in logger.handlers[:]:  # remove all old handlers
        logger.removeHandler(handler)

    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


class AverageMeter(object):
    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)