import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision
from .base import DataBase
import torch
import timm

class CognitiveData(DataBase):
    def __init__(self, cfg, **kwargs):
        """
        init data for cognitive task
        we first load the dataset and then define the transform for the dataset
        args:
            cfg: the config file
        
        cfg args:
            data_root: the root path of the dataset
            dataset: the dataset name
            batch_size: the batch size
            num_workers: the number of workers
            
        """
        super(CognitiveData, self).__init__(cfg, **kwargs)
        self.root = getattr(self.cfg, 'data_root', './data')
        # TODO: need to alternate the path to data
        self.dataset = getattr(self.cfg, 'dataset', 'cifar10')

    # TODO: change the pre-processing procedures of the dataset
    @property
    def train_transform(self):
        train_transform = {
            'cifar10': transforms.Compose(
                                                 [
                                                     transforms.RandomCrop(32, padding=4),
                                                     transforms.RandomHorizontalFlip(),
                                                     transforms.ToTensor(),
                                                     transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                          (0.2023, 0.1994, 0.2010)),
                                                 ]),
            'cifar100': transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(15),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                                         (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
                ]),
            'mnist': transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307), (0.3081))
                ]),
        }
        return train_transform[self.dataset]

    @property
    def val_transform(self):
        test_transform = {
            'cifar10': transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ]),
            'cifar100': transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                                         (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),
                ]),
            'mnist': transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307), (0.3081))
                ]),
        }
        return test_transform[self.dataset]

    # TODO: change the dataset class
    @property
    def data_cls(self):
        data_cls = {
            'cifar10': datasets.CIFAR10,
            'cifar100': datasets.CIFAR100,
            'mnist': datasets.MNIST,
        }
        return data_cls[self.dataset]

    # TODO: change these based on the dataset classes
    @property
    def train_dataset(self):
        return self.data_cls(self.root, train=True, download=True, transform=self.train_transform)

    @property
    def val_dataset(self):
        return self.data_cls(self.root, train=False, download=True, transform=self.val_transform)

    @property
    def test_dataset(self):
        return self.data_cls(self.root, train=False, download=True, transform=self.val_transform)