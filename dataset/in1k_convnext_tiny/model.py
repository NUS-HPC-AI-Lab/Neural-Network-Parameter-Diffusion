# model.py

import timm
import torch.nn as nn


class model_maker(nn.Module):
    def __init__(self, num_classes=1000):  # 修改为ImageNet的1000个类别
        super(model_maker, self).__init__()
        self.model = timm.create_model('convnext_tiny', pretrained=True, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)


def create_model(num_classes=1000):  # 修改为ImageNet的1000个类别
    return model_maker(num_classes)


# def freeze_model_except_last_two_norm(model):
#     # 打印所有的模型参数名称
#     print("Model parameter keys:")
#     for name, _ in model.named_parameters():
#         print(name)
#
#     # import pdb; pdb.set_trace()
#     print("\nFreezing parameters:")
#     for name, param in model.named_parameters():
#         if 'norm' not in name or 'stages.3' not in name:
#             param.requires_grad = False
#             print(f"Freezing: {name}")
#         else:
#             print(f"Not freezing: {name}")
#
#     print("\nTrainable parameters:")
#     for name, param in model.named_parameters():
#         if param.requires_grad:
#             print(name)
