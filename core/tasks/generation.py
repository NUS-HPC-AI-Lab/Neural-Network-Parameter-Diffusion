
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

class IGTask(BaseTask):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)