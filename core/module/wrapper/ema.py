import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from copy import deepcopy

class EMA():
    def __init__(self, model):
        self.model = model
        self.ema = deepcopy(model)