import json
from pathlib import Path

import torch
from PIL import Image

from utils import load_captions_maybe
from lavis.datasets.datasets.base_dataset import BaseDataset
from packg.iotools.jsonext import load_json
from utils import make_list_smaller
import logging

QUESTION_PROMPTS = {
    # generic
    "none": "",
    "a-photo-of": "a photo of",
    "an-image-of": "an image of",
    "what-is-this": "What is this?",
    "whats-this": "What's this?",
    "what-seen-image": "What can be seen in the image?",
    "what-is-in-image": "What is in the image?",
    "whats-in-image": "What's in the image?",
    "describe-image": "Describe the image",
    # object
    "what-object-is-this": "What object is this?",
    "what-kind-object-is-this": "What kind of object is this?",
    "what-type-object-is-this": "What type of object is this?",
    "what-class-object-is-this": "What class of object is this?",
    "what-specific-object-is-this": "What specific object is this?",
    # activity
    "what-act-is-this": "What activity is this?",
    "what-is-person-doing": "What is the person doing?",
    "what-are-people-doing": "What are the people doing?",
    "what-is-happening": "What is happening?",
    "what-is-happening-image": "What is happening in the image?",
}


def load_classif_ann(ann_paths, config):
    class_name_key = config.get("class_name_key", "class_name")
    cropped_images_dir = config.get("cropped_images_dir", "")
    annotation_dict = json.load(open(ann_paths[0]))
    # {"val_00000001": {'class_num': 489, 'image': 'val/ILSVRC2012_val_00000001.JPEG'}}
    # class_num in [0, 999] and matches answer_list class names

    annotation = []
    for key, ann in annotation_dict.items():
        if cropped_images_dir != "":
            # update image paths to point to the cropped files
            # old_path: val/ILSVRC2012_val_00000001.JPEG
            # key: val_00000001
            ann["image"] = (Path(cropped_images_dir) / ann["image"]).as_posix()
        annotation.append({"key": key, **ann})

    answer_list_path = ann_paths[1]
    classes_data = json.load(open(answer_list_path, "r", encoding="utf-8"))

    answer_list = []
    for class_data in classes_data:
        a = class_data[class_name_key]
        # a = text_processor_fn(a)  # convert to lowercase for blip-like models
        answer_list.append(a)

    if cropped_images_dir != "":
        print(f"Updated paths to use cropped images from '{cropped_images_dir}'")

    # list of str len 1000
    return annotation, answer_list


class ClassifierVQADataset(BaseDataset):
    _key_field: str = "question_id"

    # noinspection PyMissingConstructor
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, config):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_paths (string): list of annotation paths
        config: The "dataset" part of the config e.g.
            {'data_type': 'images', 'build_info': ..., 'annotations': ...,
            'type': 'eval', 'vis_processor': ..., 'text_processor': ...,
            'debug_max': 100, # <- note the -d options appear here
            }

        """
        self.config = config
        self.vis_root = vis_root
        self.vis_processor = vis_processor
        self.text_processor = text_processor
        self.annotation, self.answer_list = load_classif_ann(ann_paths, config)
        self._add_instance_ids(key=self._key_field)  # question_id unique

        instance_ids = [int(anno[self._key_field]) for anno in self.annotation]
        self.question_captions = load_captions_maybe(config, instance_ids)
        self.return_visual = config.get("return_visual", True)

        # some of the code is using "answer_list" as classes and some "classnames"
        # so we add both
        self.classnames = self.answer_list
        # by default it uses the openai imagenet prompt templates
        # if needed change this for the specific dataset
        self.classtemplates = "openai_imagenet_template"

        self.classsynonyms = None
        synonyms_file = config.get("synonyms")
        if synonyms_file is not None:
            synonyms_dict = load_json(synonyms_file)

            # make sure the dict is sorted like {"person": 0, "human": 0, "dog": 1, ...}
            classids = list(synonyms_dict.values())
            assert classids == sorted(classids)
            set_classids = set(classids)
            num_classes = len(set_classids)
            assert sorted(set_classids) == list(range(num_classes))
            assert num_classes == len(self.classnames)

            # MultimodalClassificationSynonymsTask expects a list of list
            classsynonyms = [[] for _ in range(num_classes)]
            for synonym, classid in synonyms_dict.items():
                classsynonyms[classid].append(synonym)
            self.classsynonyms = classsynonyms

    def __getitem__(self, index):
        ann = self.annotation[index]
        class_idx = int(ann["class_idx"])
        class_name = self.classnames[class_idx]
        sample = {
            "question_id": int(ann[self._key_field]),  # needed for vqa task
            "image_id": int(ann[self._key_field]),  # needed for captioning task
            "instance_id": int(ann[self._key_field]),  # needed for multimodalcls task
            "class_idx": class_idx,
            "class_name": class_name,
            "image_file": ann["image"],
            "label": class_idx,
            "multiple_choice_answer": class_name,
        }
        cropped_images_dir = self.config.get("cropped_images_dir", "")
        if self.return_visual:
            if cropped_images_dir == "":
                image_path = Path(self.vis_root) / ann["image"]
            # print(f"Image path: {image_path}")
            image_pil = Image.open(image_path).convert("RGB")
            sample["image_raw"] = image_pil

        # add object to ask followup questions about in case it is in the annotation
        question = self.config.get("question_type", "none")
        question_followup = ann.get("question_followup", None)

        if question_followup is not None:
            # task is classifier_vqa_followup, ask the followup question e.g. what type of dog?
            sample["text_input_raw"] = question_followup
        elif question not in {"none", ""}:
            # task is classifier_vqa, ask the classifier question e.g. what is in the image?
            sample["text_input_raw"] = QUESTION_PROMPTS[question]

        # # captions are for pnpvqa
        if self.question_captions is not None:
            # qids in the captions start counting as 0
            # image ids of imagenet start counting as 1
            qid = int(ann[self._key_field])
            sample["captions"] = self.question_captions[qid]
            # print(f"Captions: {sample['captions']} for question id: {qid}")

        return sample

    def collater(self, samples):
        # Filter out None samples
        samples = [s for s in samples if s is not None]
        # Check if samples is empty after filtering
        if not samples:
            return None
        image_raw_list, question_raw_list, multiple_choice_answer_list = [], [], []
        question_id_list, image_file_list = [], []

        for sample in samples:
            image_raw_list.append(sample["image_raw"])
            question_raw_list.append(sample["text_input_raw"])
            multiple_choice_answer_list.append(sample["multiple_choice_answer"])

            question_id_list.append(sample["question_id"])
            image_file_list.append(sample["image_file"])

        return {
            "image_raw": image_raw_list,
            "text_input_raw": question_raw_list,
            "multiple_choice_answer": multiple_choice_answer_list,
            "question_id": question_id_list,
            "file": image_file_list,
        }