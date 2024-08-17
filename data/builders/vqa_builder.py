"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder

from lavis.common.registry import registry
from data.coco_vqa import *
from data.classifier_vqa_dataset import ClassifierVQADataset
from pathlib import Path
import warnings


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

@registry.register_builder("imagenet1k")
class ImagenetVQABuilder(BaseDatasetBuilder):
    train_dataset_cls = ClassifierVQADataset
    eval_dataset_cls = ClassifierVQADataset

    DATASET_CONFIG_DICT = {
        "default": "/nethome/chuang475/flash/projects/vlm_robustness/data/configs/imagenet1k.yaml",
        "eval": "/nethome/chuang475/flash/projects/vlm_robustness/data/configs/imagenet1k.yaml",
    }
    
    def build(self):
        """
        Create by split datasets inheriting torch.utils.data.Datasets.

        # build() can be dataset-specific. Overwrite to customize.
        """
        self.build_processors()

        build_info = self.config.build_info

        ann_info = build_info.annotations
        vis_info = build_info.get(self.data_type)

        datasets = dict()
        for split in ann_info.keys():
            # # change: removed this to allow custom splits
            # if split not in ["train", "val", "test"]:
            #     continue

            is_train = split == "train"

            # processors
            vis_processor = (
                self.vis_processors["train"] if is_train else self.vis_processors["eval"]
            )
            text_processor = (
                self.text_processors["train"] if is_train else self.text_processors["eval"]
            )

            # annotation path
            ann_paths = ann_info.get(split).storage
            if isinstance(ann_paths, str):
                ann_paths = [ann_paths]

            # check if ann_paths are real paths or relative to cache dir
            new_ann_paths = []
            for ann_path in ann_paths:
                ann_path = Path(ann_path)
                if not ann_path.is_file():
                    ann_path_new = get_ovqa_cache_dir() / ann_path
                    if not ann_path_new.is_file():
                        raise FileNotFoundError(
                            f"Could not find either {ann_path.as_posix()} or {ann_path_new.as_posix()}"
                        )
                    ann_path = ann_path_new
                new_ann_paths.append(ann_path)
            ann_paths = new_ann_paths

            # visual data storage path
            vis_path = vis_info.storage
            if not os.path.exists(vis_path):
                warnings.warn("storage path {} does not exist.".format(vis_path))

            # create datasets
            dataset_cls = self.train_dataset_cls if is_train else self.eval_dataset_cls
            self.config.build_info.annotations[split]["split"] = split

            datasets[split] = dataset_cls(
                vis_processor=vis_processor,
                text_processor=text_processor,
                ann_paths=ann_paths,
                vis_root=vis_path,
                config=self.config,
            )

        return datasets

@registry.register_builder("imagenet-2")
class Imagenet2VQABuilder(ImagenetVQABuilder):
    train_dataset_cls = None
    eval_dataset_cls = ClassifierVQADataset

    DATASET_CONFIG_DICT = {
        "default": "",
        "eval": "/nethome/chuang475/flash/projects/vlm_robustness/data/configs/imagenet-2.yaml",
    }

@registry.register_builder("imagenet-a")
class ImagenetAVQABuilder(ImagenetVQABuilder):
    train_dataset_cls = None
    eval_dataset_cls = ClassifierVQADataset

    DATASET_CONFIG_DICT = {
        "default": "",
        "eval": "/nethome/chuang475/flash/projects/vlm_robustness/data/configs/imagenet-a.yaml",
    }

@registry.register_builder("imagenet-r")
class ImagenetRVQABuilder(ImagenetVQABuilder):
    train_dataset_cls = None
    eval_dataset_cls = ClassifierVQADataset

    DATASET_CONFIG_DICT = {
        "default": "",
        "eval": "/nethome/chuang475/flash/projects/vlm_robustness/data/configs/imagenet-r.yaml",
    }

@registry.register_builder("imagenet-s")
class ImagenetSVQABuilder(ImagenetVQABuilder):
    train_dataset_cls = None
    eval_dataset_cls = ClassifierVQADataset

    DATASET_CONFIG_DICT = {
        "default": "",
        "eval": "/nethome/chuang475/flash/projects/vlm_robustness/data/configs/imagenet-s.yaml",
    }