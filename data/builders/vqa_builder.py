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
        "default": "/coc/pskynet4/chuang475/projects/vlm_robustness/data/configs/defaults_vqa_raw.yaml",
        "eval": "/coc/pskynet4/chuang475/projects/vlm_robustness/data/configs/eval_vqa_raw.yaml",
    }


@registry.register_builder("coco_vqa_cp")
class COCOVQACPBuilder(BaseDatasetBuilder):
    train_dataset_cls = COCOVQADataset_Raw
    eval_dataset_cls = COCOVQAEvalDataset_Raw

    DATASET_CONFIG_DICT = {
        "default": "/coc/pskynet4/chuang475/projects/vlm_robustness/data/configs/vqa_cp.yaml"
    }


@registry.register_builder("coco_vqa_rephrasings")
class COCOVQA_Rephrasings_Builder(BaseDatasetBuilder):
    train_dataset_cls = COCOVQADataset_Raw
    eval_dataset_cls = COCOVQAEvalDataset_Raw

    DATASET_CONFIG_DICT = {
        "default": "/coc/pskynet4/chuang475/projects/vlm_robustness/data/configs/vqa_rephrasings.yaml"
    }

@registry.register_builder("coco_vqa_lol")
class COCOVQALOLBuilder(BaseDatasetBuilder): 
    train_dataset_cls = COCOVQADataset_Raw
    eval_dataset_cls = COCOVQAEvalDataset_Raw

    DATASET_CONFIG_DICT = { 
        "default": "/coc/pskynet4/bmaneech3/vlm_robustness/data/configs/vqa_lol.yaml"
    }

@registry.register_builder("coco_vqa_vs")
class COCOVQAVSBuilder(BaseDatasetBuilder): 

    train_dataset_cls = COCOVQADataset_Raw
    eval_dataset_cls = COCOVQAEvalDataset_Raw

    DATASET_CONFIG_DICT = { 
        "default": "/coc/pskynet4/bmaneech3/vlm_robustness/data/configs/vqa_vs.yaml"
    }

@registry.register_builder("coco_vqa_ce")
class COCOVQACEBuilder(BaseDatasetBuilder): 
    train_dataset_cls = COCOVQADataset_Raw
    eval_dataset_cls = COCOVQAEvalDataset_Raw

    DATASET_CONFIG_DICT = { 
        "default": "/coc/pskynet4/chuang475/projects/vlm_robustness/data/configs/vqa_ce.yaml"
    }

@registry.register_builder("coco_cv-vqa")
class COCOCVVQABuilder(BaseDatasetBuilder): 
    train_dataset_cls = COCOVQADataset_Raw
    eval_dataset_cls = COCOVQAEvalDataset_Raw

    DATASET_CONFIG_DICT = { 
        "default": "/coc/pskynet4/chuang475/projects/vlm_robustness/data/configs/cv-vqa.yaml"
    }

@registry.register_builder("coco_iv-vqa")
class COCOIVVQABuilder(BaseDatasetBuilder): 
    train_dataset_cls = COCOVQADataset_Raw
    eval_dataset_cls = COCOVQAEvalDataset_Raw

    DATASET_CONFIG_DICT = { 
        "default": "/coc/pskynet4/chuang475/projects/vlm_robustness/data/configs/iv-vqa.yaml"
    }

@registry.register_builder("coco_advqa")
class COCOADVQABuilder(BaseDatasetBuilder): 
    train_dataset_cls = COCOVQADataset_Raw
    eval_dataset_cls = COCOVQAEvalDataset_Raw

    DATASET_CONFIG_DICT = { 
        "default": "/coc/pskynet4/chuang475/projects/vlm_robustness/data/configs/advqa.yaml"
    }

@registry.register_builder("textvqa")
class TextVQABuilder(BaseDatasetBuilder): 
    train_dataset_cls = COCOVQADataset_Raw
    eval_dataset_cls = COCOVQAEvalDataset_Raw

    DATASET_CONFIG_DICT = { 
        "default": "/coc/pskynet4/chuang475/projects/vlm_robustness/data/configs/textvqa.yaml"
    }

@registry.register_builder("vizwiz")
class VizWizBuilder(BaseDatasetBuilder): 
    train_dataset_cls = COCOVQADataset_Raw
    eval_dataset_cls = COCOVQAEvalDataset_Raw

    DATASET_CONFIG_DICT = { 
        "default": "/coc/pskynet4/chuang475/projects/vlm_robustness/data/configs/vizwiz.yaml"
    }

@registry.register_builder("coco_okvqa")
class COCOOKVQABuilder(BaseDatasetBuilder): 
    train_dataset_cls = COCOVQADataset_Raw
    eval_dataset_cls = COCOVQAEvalDataset_Raw

    DATASET_CONFIG_DICT = { 
        "default": "/coc/pskynet4/chuang475/projects/vlm_robustness/data/configs/ok-vqa.yaml"
    }