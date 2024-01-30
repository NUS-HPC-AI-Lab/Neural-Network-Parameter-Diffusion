import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from copy import deepcopy

class EMA():
    def __init__(self, model):
        self.model = model.cuda()
        self.ema = deepcopy(model)

    def parameters(self):
        return self.model.parameters()

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)