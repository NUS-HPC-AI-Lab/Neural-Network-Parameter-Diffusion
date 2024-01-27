import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision
from .base import DataBase
import torch
import timm


class VisionData(DataBase):
    def __init__(self, cfg, **kwargs):
        super(VisionData, self).__init__(cfg, **kwargs)

        """
        init data for classification task
        we firstly load the dataset and then define the transform for the dataset
        args:
            cfg: the config file

        cfg args:
            data_root: the root path of the dataset
            dataset: the dataset name
            batch_size: the batch size
            num_workers: the number of workers

        """
        super(VisionData, self).__init__(cfg, **kwargs)
        self.root = getattr(self.cfg, 'data_root', './data')
        self.dataset = getattr(self.cfg, 'dataset', 'cifar10')

    @property
    def train_transform(self):
        train_transform = {
        }
        return train_transform[self.dataset]

    @property
    def val_transform(self):
        test_transform = {
        }
        return test_transform[self.dataset]

    @property
    def data_cls(self):
        data_cls = {

        }
        return data_cls[self.dataset]

    @property
    def train_dataset(self):
        return self.data_cls(self.root, train=True, download=True, transform=self.train_transform)

    @property
    def val_dataset(self):
        return self.data_cls(self.root, train=False, download=True, transform=self.val_transform)

    @property
    def test_dataset(self):
        return self.data_cls(self.root, train=False, download=True, transform=self.val_transform)