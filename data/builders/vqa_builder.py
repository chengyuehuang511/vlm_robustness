"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder

from lavis.common.registry import registry
from data.coco_vqa import *


@registry.register_builder("coco_vqa_raw")
class COCOVQABuilder_Raw(BaseDatasetBuilder):
    train_dataset_cls = COCOVQADataset_Raw
    eval_dataset_cls = COCOVQAEvalDataset_Raw

    DATASET_CONFIG_DICT = {
        "default": "/nethome/chuang475/flash/projects/vlm_robustness/data/configs/defaults_vqa_raw.yaml",
        "eval": "/nethome/chuang475/flash/projects/vlm_robustness/data/configs/eval_vqa_raw.yaml",
    }


@registry.register_builder("coco_vqa_cp")
class COCOVQACPBuilder(BaseDatasetBuilder):
    train_dataset_cls = COCOVQADataset_Raw
    eval_dataset_cls = COCOVQAEvalDataset_Raw

    DATASET_CONFIG_DICT = {
        "default": "/nethome/chuang475/flash/projects/vlm_robustness/data/configs/vqa_cp.yaml"
    }

@registry.register_builder("coco_vqa_rephrasings")
class COCOVQA_Rephrasings_Builder(BaseDatasetBuilder):
    train_dataset_cls = COCOVQADataset_Raw
    eval_dataset_cls = COCOVQAEvalDataset_Raw

    DATASET_CONFIG_DICT = {
        "default": "/nethome/chuang475/flash/projects/vlm_robustness/data/configs/vqa_rephrasings.yaml"
    }