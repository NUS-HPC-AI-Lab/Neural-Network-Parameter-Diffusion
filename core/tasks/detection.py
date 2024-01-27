import hydra.utils

from .base_task import  BaseTask
from core.data.vision_dataset import VisionData
from core.data.parameters import PData
from torchvision.io.image import read_image
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image


# object detection task
class ODTask(BaseTask):
    def __init__(self, config, **kwargs):
        super(ODTask, self).__init__(config, **kwargs)
        self.train_loader = self.task_data.train_dataloader()
        self.eval_loader = self.task_data.val_dataloader()
        self.test_loader = self.task_data.test_dataloader()

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
        return

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