# model.py

import timm
import torch.nn as nn


class convnext_tiny(nn.Module):
    def __init__(self, num_classes=100):
        super(convnext_tiny, self).__init__()
        self.model = timm.create_model('convnext_tiny', pretrained=True, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)


def create_model(num_classes=100):
    return convnext_tiny(num_classes)
