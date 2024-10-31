# model.py

import timm
import torch.nn as nn


class resnet50(nn.Module):
    def __init__(self, num_classes=100):
        super(resnet50, self).__init__()
        self.model = timm.create_model('resnet50', pretrained=True, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)


def create_model(num_classes=100):
    return resnet50(num_classes)
