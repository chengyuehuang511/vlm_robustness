
"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import argparse
# import os
# print(os.environ["CUBLAS_WORKSPACE_CONFIG"])
# print(os.environ["PYTHONHASHSEED"])

import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import logging
import yaml
from lavis.common.dist_utils import get_rank, init_distributed_mode

def setup_seeds(seed):
    seed = seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

    cudnn.benchmark = False
    cudnn.deterministic = True

    torch.use_deterministic_algorithms(True)

setup_seeds(42)

import lavis.tasks as tasks
from lavis.common.config import Config
from lavis.common.logger import setup_logger
from lavis.common.optims import (
    LinearWarmupCosineLRScheduler,
    LinearWarmupStepLRScheduler,
)
from lavis.common.registry import registry
from lavis.common.utils import now

# imports modules for registration
from lavis.datasets.builders import *
from lavis.models import *
from data.builders import *
from model import *
from optimizer import *
from runners import *
from tasks import *

from lavis.processors import *
from lavis.runners import *

def get_runner_class(cfg):
    """
    Get runner class from config. Default to epoch-based runner.
    """
    runner_cls = registry.get_runner_class(cfg.run_cfg.get("runner", "runner_robust_ft"))  # runner_base  # runner_robust_ft

    return runner_cls

def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )

    args = parser.parse_args()
    # if 'LOCAL_RANK' not in os.environ:
    #     os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


job_id = now()

# args = parse_args()
# cfg = Config(args)


cfg = Config(parse_args())
init_distributed_mode(cfg.run_cfg)

# setup_seeds(cfg)

# set after init_distributed_mode() to only log on master.
setup_logger()

cfg.pretty_print()

task = tasks.setup_task(cfg)

datasets = task.build_datasets(cfg)
model = task.build_model(cfg)

runner = get_runner_class(cfg)( #robust FT runner class
        cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets
    )

runner.get_hidden_states(skip_reload=True)
print(model)




"""
steps : 
1) write new output hidden states function to get from model 
2) return and find way to combine between the multiple batches 
3) 

"""