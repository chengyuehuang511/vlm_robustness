"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from lavis.common.registry import registry
from tasks.classifier_vqa_task import ClassifierVQATask
from tasks.classifier_vqa_followup_task import ClassifierVQAFollowupTask
# registry.unregister("vqa")
from tasks.vqa import VQATask


def setup_task(cfg):
    assert "task" in cfg.run_cfg, "Task name must be provided."

    task_name = cfg.run_cfg.task
    task = registry.get_task_class(task_name).setup_task(cfg=cfg)
    assert task is not None, "Task {} not properly registered.".format(task_name)

    return task


__all__ = [
    "ClassifierVQATask",
    "ClassifierVQAFollowupTask",
    "VQATask",
]