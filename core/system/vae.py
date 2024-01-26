import hydra.utils
import torch

from .base import BaseSystem
import torch.nn as nn
import pytorch_lightning as pl
import pdb


class VAESystem(BaseSystem):
    def __init__(self, config, **kwargs):
        super(VAESystem, self).__init__(config)
        print("VAESystem init")


    def encode(self, x, **kwargs):
        return self.model.encode(x)


    def decode(self, x, **kwargs):
        return self.model.decode(x)