import hydra.utils

from .base_task import  BaseTask
from core.data.vision_dataset import VisionData
from core.data.parameters import PData
from core.utils.utils import *
import torch.nn as nn
import datetime
from core.utils import *
import glob
import omegaconf
import json


# object detection task
class ODTask(BaseTask):
    def __init__(self, config, **kwargs):
        super(ODTask, self).__init__(config, **kwargs)
        self.train_loader = self.task_data.train_dataloader()
        self.eval_loader = self.task_data.val_dataloader()
        self.test_loader = self.task_data.test_dataloader()

    def init_task_data(self):
        pass

    def set_param_data(self):
        pass

    def test_g_model(self, input):
        pass

    def train_for_data(self):
        pass

    def train(self):
        pass

    def test(self):
        pass