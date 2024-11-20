"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import json
import random
from PIL import Image
import torch

from lavis.datasets.datasets.vqa_datasets import VQADataset, VQAEvalDataset
from lavis.datasets.datasets.base_dataset import BaseDataset

from collections import OrderedDict


class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict(
            {
                "file": ann["image"],
                "question": ann["question"],
                "question_id": ann["question_id"],
                "answers": "; ".join(ann["answer"]),
                "image": sample["image"],
            }
        )


class COCOVQADataset(VQADataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):
        ann = self.annotation[index]

        try : 
            image_path = os.path.join(self.vis_root, ann["image"])


        except Exception as e : 
            print(type(ann))
            print(ann)
            raise e
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        question = self.text_processor(ann["question"])

        answer_weight = {}
        for answer in ann["answer"]:
            if answer in answer_weight.keys():
                answer_weight[answer] += 1 / len(ann["answer"])
            else:
                answer_weight[answer] = 1 / len(ann["answer"])

        answers = list(answer_weight.keys())
        weights = list(answer_weight.values())

        return {
            "image": image,
            "text_input": question,
            "answers": answers,
            "weights": weights,
        }


# 1
class VQADataset_Raw(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def collater(self, samples):
        # Filter out None samples
        samples = [s for s in samples if s is not None]
        # Check if samples is empty after filtering
        if not samples:
            return None
        answer_list, weight_list = [], []
        image_raw_list, question_raw_list, multiple_choice_answer_list = [], [], []

        num_answers = []

        for sample in samples:
            image_raw_list.append(sample["image_raw"])
            question_raw_list.append(sample["text_input_raw"])

            multiple_choice_answer_list.append(sample["multiple_choice_answer"])

            weight_list.extend(sample["weights"])

            answers = sample["answers"]

            answer_list.extend(answers)
            num_answers.append(len(answers))

        return {
            "image_raw": image_raw_list,
            "text_input_raw": question_raw_list,
            "answer": answer_list,
            "weight": weight_list,
            "n_answers": torch.LongTensor(num_answers),
            "multiple_choice_answer": multiple_choice_answer_list,
        }
    

# 2
class COCOVQADataset_Raw(VQADataset_Raw, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):
        ann = self.annotation[index]
        # print(ann)

        image_path = os.path.join(self.vis_root, ann["image"])
        image_raw = Image.open(image_path).convert("RGB")

        answer_weight = {}
        for answer in ann["answer"]:
            if answer in answer_weight.keys():
                answer_weight[answer] += 1 / len(ann["answer"])
            else:
                answer_weight[answer] = 1 / len(ann["answer"])

        answers = list(answer_weight.keys())
        weights = list(answer_weight.values())

        # select the most frequent multiple_choice_answer in the list - ann["answer"]
        multiple_choice_answer = max(set(ann["answer"]), key=ann["answer"].count)

        return {
            "answers": answers,
            "multiple_choice_answer": multiple_choice_answer,
            "weights": weights,
            "image_raw": image_raw,
            "text_input_raw": ann["question"],
        }


class COCOVQAInstructDataset(COCOVQADataset):
    def __getitem__(self, index):
        data = super().__getitem__(index)
        if data != None:
            data['text_output'] = random.choice(data["answers"])
        return data

    def collater(self, samples):
        data = super().collater(samples)
        data['text_output'] = data['answer']
        return data

    

class COCOVQAEvalDataset(VQAEvalDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """

        self.vis_root = vis_root

        self.annotation = json.load(open(ann_paths[0]))
        print("-------------------",vis_root)
        print("-------------------",ann_paths)
        print("------------1", self.annotation[0])
        print("------------------- keys", self.annotation[0].keys())
        # print(self.annotation[0])

        answer_list_path = ann_paths[1]
        if os.path.exists(answer_list_path):
            self.answer_list = json.load(open(answer_list_path))
            # print("-------------2", self.answer_list[0])
        else:
            self.answer_list = None

        try:
            self.coco_fmt_qust_file = ann_paths[2]
            self.coco_fmt_anno_file = ann_paths[3]
        except IndexError:
            self.coco_fmt_qust_file = None
            self.coco_fmt_anno_file = None

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        # print("------------3", self.coco_fmt_qust_file)
        # print("------------4", self.coco_fmt_anno_file)

        self._add_instance_ids()

    def __getitem__(self, index):
        ann = self.annotation[index]
        # print("------------------", ann)

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        question = self.text_processor(ann["question"])

        return {
            "image": image,
            "text_input": question,
            "question_id": ann["question_id"],
            "instance_id": ann["instance_id"],
        }


# 3
class COCOVQAEvalDataset_Raw(VQAEvalDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """

        self.vis_root = vis_root
        # fix_path = fix_path.replace("/nethome/chuang475/flash", "/coc/pskynet4/chuang475")

        for i in range(len(ann_paths)) : 
            ann_paths[i] = ann_paths[i].replace("/nethome/chuang475/flash", "/coc/pskynet4/chuang475")

        self.annotation = json.load(open(ann_paths[0]))

        answer_list_path = ann_paths[1]
        if os.path.exists(answer_list_path):
            self.answer_list = json.load(open(answer_list_path))
        else:
            self.answer_list = None

        try:
            self.coco_fmt_qust_file = ann_paths[2]
            self.coco_fmt_anno_file = ann_paths[3]
        except IndexError:
            self.coco_fmt_qust_file = None
            self.coco_fmt_anno_file = None

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self._add_instance_ids()

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image_raw = Image.open(image_path).convert("RGB")
        multiple_choice_answer = max(set(ann["answer"]), key=ann["answer"].count)

        return {
            "question_id": ann["question_id"],
            "instance_id": ann["instance_id"],
            "image_raw": image_raw,
            "image_path": image_path,
            "text_input_raw": ann["question"],
            "multiple_choice_answer": multiple_choice_answer,
        }
    

# class COCOVQALOLDataset(VQADataset, __DisplMixin) : 
#     def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
#         super().__init__(vis_processor, text_processor, vis_root, ann_paths)

#     def __getitem__(self, index):
#         ann = self.annotation[index]

#         split = ann["img_id"].split('_')[1]
#         image_path = os.path.join(self.vis_root, split, f"{ann['img_id']}.jpg")
#         image = Image.open(image_path).convert("RGB")

#         image = self.vis_processor(image)
#         question = self.text_processor(ann["sent"])

        

#         answer_weight = {}
#         total_sum = sum(value for value in ann["label"].values() if value is not None)
#         for answer in ann["label"]:
#             if ann["label"][answer]== None: 
#                 continue 
#             answer_weight[answer] = ann["label"][answer] / total_sum

#         answers = list(answer_weight.keys())
#         weights = list(answer_weight.values())
#         return {
#             "image": image,
#             "text_input": question,
#             "answers": answers,
#             "weights": weights,
#         }



# class COCOVQALOLEvalDataset(VQAEvalDataset,  __DisplMixin) : 
#     def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
#         """
#         vis_root (string): Root directory of images (e.g. coco/images/)
#         ann_root (string): directory to store the annotation file
#         """

#         self.vis_root = vis_root

#         self.annotation = json.load(open(ann_paths[1])) #create vqa_lol file that combines all values 

#         answer_list_path = ann_paths[0]
#         if os.path.exists(answer_list_path):
#             self.answer_list = json.load(open(answer_list_path))
#         # else:
#         #     self.answer_list = None

#         # try:
#         #     self.coco_fmt_qust_file = ann_paths[1]
#         #     self.coco_fmt_anno_file = ann_paths[1]
#         # except IndexError:
#         self.coco_fmt_qust_file = None
#         self.coco_fmt_anno_file = None

#         self.vis_processor = vis_processor
#         self.text_processor = text_processor

#         self._add_instance_ids()


#     def __getitem__(self, index):
#         ann = self.annotation[index]


#         split = ann["img_id"].split('_')[1]
#         image_path = os.path.join(self.vis_root, split, f"{ann['img_id']}.jpg")
#         # print("IMAGE PATH", image_path)
#         image_raw = Image.open(image_path).convert("RGB") 

#         image = self.vis_processor(image_raw)
#         question = self.text_processor(ann["sent"])

#         return {
#             "image": image,
#             "text_input": question,
#             "question_id": ann["question_id"],
#             "instance_id": ann["instance_id"],
#             "image_raw": image_raw,
#             "text_input_raw": ann["sent"],
#         }




#write code to combine the 