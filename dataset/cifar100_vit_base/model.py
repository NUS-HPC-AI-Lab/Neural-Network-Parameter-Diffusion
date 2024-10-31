# model.py

import timm
import torch.nn as nn


class convnext_base(nn.Module):
    def __init__(self, num_classes=100):
        super(convnext_base, self).__init__()
        self.model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)


def create_model(num_classes=100):
    return convnext_base(num_classes)
