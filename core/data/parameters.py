import torch.nn as nn
from torchvision.datasets.vision import VisionDataset
import os
import torch
import pdb
from torchvision.datasets.utils import check_integrity, download_and_extract_archive
from .base import DataBase
from torch.utils.data import Dataset
import warnings
import os

class PData(DataBase):
    def __init__(self, cfg, **kwargs):
        super(PData, self).__init__(cfg, **kwargs)
        self.root = getattr(self.cfg, 'data_root', './data')
        self.k = getattr(self.cfg, 'k', 200)
        self.batch_size = self.k

        # check the root path is  exist or not
        assert os.path.exists(self.root), f'{self.root} not exists'

        # check the root is a directory or file
        if os.path.isfile(self.root):
            state = torch.load(self.root, map_location='cpu')
            self.fix_model = state['model']
            self.fix_model.eval()
            self.fix_model.to('cpu')
            self.fix_model.requires_grad_(False)

            self.pdata = state['pdata']
            self.accuracy = state['performance']
            self.train_layer = state['train_layer']

        elif os.path.isdir(self.root):
            pass

    def get_train_layer(self):
        return self.train_layer

    def get_model(self):
        return self.fix_model

    def get_accuracy(self):
        return self.accuracy

    @property
    def train_dataset(self):
        return Parameters(self.pdata, self.k, split='train')

    @property
    def val_dataset(self):
        return Parameters(self.pdata, self.k, split='val')

    @property
    def test_dataset(self):
        return Parameters(self.pdata, self.k, split='test')


class Parameters(VisionDataset):
    def __init__(self, batch, k, split='train'):
        super(Parameters, self).__init__(root=None, transform=None, target_transform=None)
        if split  == 'train':
            self.data = batch[:k]
        else:
            self.data = batch[:k]
        # data is a tensor list which is the parameters of the model

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self) -> int:
        return len(self.data)


