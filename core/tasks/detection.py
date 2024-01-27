import hydra.utils

from .base_task import  BaseTask
from core.data.vision_dataset import VisionData
from core.data.parameters import PData
from torchvision.io.image import read_image
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image

import datetime
import os
import pdb
import time
import object_detection.presets
import torch
import torch.utils.data
import torchvision
import torchvision.models.detection
import torchvision.models.detection.mask_rcnn
import object_detection.utils
from object_detection.coco_utils import get_coco
from object_detection.engine import evaluate, train_one_epoch
from object_detection.group_by_aspect_ratio import create_aspect_ratio_groups, GroupedBatchSampler
from torchvision.transforms import InterpolationMode
from object_detection.transforms import SimpleCopyPaste
from object_detection.train import *


# object detection task
class ODTask(BaseTask):
    def __init__(self, config, **kwconfig):
        super(ODTask, self).__init__(config, **kwconfig)
        if self.cfg.backend.lower() == "tv_tensor" and not self.cfg.use_v2:
            raise ValueError("Use --use-v2 if you want to use the tv_tensor backend.")
        if self.cfg.dataset not in ("coco", "coco_kp"):
            raise ValueError(f"Dataset should be coco or coco_kp, got {config.dataset}")
        if "keypoint" in config.model and self.cfg.dataset != "coco_kp":
            raise ValueError("Oops, if you want Keypoint detection, set --dataset coco_kp")
        if self.cfg.dataset == "coco_kp" and self.cfg.use_v2:
            raise ValueError("KeyPoint detection doesn't support V2 transforms yet")


    def build_model(self):
        if self.cfg.model == 'faster_rcnn_v1':
            weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
            model = fasterrcnn_resnet50_fpn(weights=weights, box_score_thresh=0.9)
        elif self.cfg.model == 'faster_rcnn_v2':
            weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
            model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.9)
        else:
            raise NotImplementedError
        return model


    def init_task_data(self):
        dataset, num_classes = get_dataset(is_train=True, args=self.cfg.data)
        dataset_test, _ = get_dataset(is_train=False, args=self.cfg.data)

        train_collate_fn = utils.collate_fn

        self.data_loader = torch.utils.data.DataLoader(
            dataset, num_workers=args.workers, collate_fn=train_collate_fn
        )

        self.data_loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=1, num_workers=args.workers, collate_fn=utils.collate_fn
        )

    def set_param_data(self):
        pass

    def test_g_model(self, input):
        pass

    def train_for_data(self):
        pass