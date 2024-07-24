"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.datasets.builders.base_dataset_builder import load_dataset_config
import os 
# print(os.getcwd())

from data.builders.vqa_builder import *

from lavis.common.registry import registry

__all__ = [
    "COCOVQACPBuilder",
    "COCOVQABuilder_Raw",
    "COCOVQAVSBuilder", 
    "COCOVQALOLBuilder"
    "COCOVQA_Rephrasings_Builder",
    "COCOVQACEBuilder",
    "COCOVQALOLBuilder",
    "COCOVQAVSBuilder",
    "COCOCVVQABuilder",
    "COCOIVVQABuilder",
    'COCOADVQABuilder',
    'TextVQABuilder',
    'VizWizBuilder',
    'COCOOKVQABuilder'
]


def load_dataset(name, cfg_path=None, vis_path=None, data_type=None):
    """
    Example

    >>> dataset = load_dataset("coco_caption", cfg=None)
    >>> splits = dataset.keys()
    >>> print([len(dataset[split]) for split in splits])

    """
    if cfg_path is None:
        cfg = None
    else:
        cfg = load_dataset_config(cfg_path)

    try:
        builder = registry.get_builder_class(name)(cfg)
    except TypeError:
        print(
            f"Dataset {name} not found. Available datasets:\n"
            + ", ".join([str(k) for k in dataset_zoo.get_names()])
        )
        exit(1)

    if vis_path is not None:
        if data_type is None:
            # use default data type in the config
            data_type = builder.config.data_type

        assert (
            data_type in builder.config.build_info
        ), f"Invalid data_type {data_type} for {name}."

        builder.config.build_info.get(data_type).storage = vis_path

    dataset = builder.build_datasets()
    return dataset


class DatasetZoo:
    def __init__(self) -> None:
        self.dataset_zoo = {
            k: list(v.DATASET_CONFIG_DICT.keys())
            for k, v in sorted(registry.mapping["builder_name_mapping"].items())
        }

    def get_names(self):
        return list(self.dataset_zoo.keys())

dataset_zoo = DatasetZoo()
# coco_vqa_vs
vqa_vs = load_dataset("coco_vqa_vs")

print(vqa_vs.keys())
print(vqa_vs["train"][0])

