import torch
from data_utils.Parameter_dataset import Parameters_partial, Parameters
from lightning import LightningDataModule

class ParametersDataModule(LightningDataModule):

    def __init__(self, data_dir='../../datasets/', batch_size=256, num_workers=8, download=True, augmentation='BYOL', lineartest=False, GaussianBlur=1, num_model=1, **kwargs):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.download = download
        self.GaussianBlur = GaussianBlur
        self.augmentation = augmentation
        self.lineartest = lineartest
        self.num_model = num_model
        # self.setup()
    @property
    def dataset(self):
        return Parameters


    @property
    def num_classes(self):
        return 10

    def prepare_data(self): # called only on 1 GPU
        if self.download:
            self.dataset(self.data_dir, train=True, download=self.download, num_model=self.num_model)

    def setup(self, stage=None): # called on every GPU
        self.train = self.dataset(self.data_dir, train=True, download=True, transform=None, num_model=self.num_model)
        self.val = self.train # self.dataset(self.data_dir, train=False, download=True, transform=None)

        self.mean = self.train.mean
        self.std = self.train.std

    def train_dataloader(self):
        print('self.batch_size,', self.batch_size)
        loader = torch.utils.data.DataLoader(
            dataset=self.train,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            drop_last=True)
        return loader

    def val_dataloader(self):
        loader = torch.utils.data.DataLoader(
            dataset=self.val,
            batch_size=self.batch_size*10,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
            drop_last=False)
        return loader



class ParametersDataModule_partial(LightningDataModule):

    def __init__(self, data_dir='../../datasets/', batch_size=256, num_workers=8, download=True, augmentation='BYOL', lineartest=False, GaussianBlur=1, num_model=1, target_layer='classifier',size='conv3',**kwargs):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.download = download
        self.GaussianBlur = GaussianBlur
        self.augmentation = augmentation
        self.lineartest = lineartest
        self.num_model = num_model
        self.target_layer = target_layer
        self.size = size
        # self.setup()
    @property
    def dataset(self):
        return Parameters_partial


    @property
    def num_classes(self):
        return 10

    def prepare_data(self): # called only on 1 GPU
        if self.download:
            self.dataset(self.data_dir, train=True, download=self.download, num_model=self.num_model, size=self.size)

    def setup(self, stage=None): # called on every GPU
        self.train = self.dataset(self.data_dir, train=True, download=True, transform=None, num_model=self.num_model, size=self.size)
        self.val = self.train # self.dataset(self.data_dir, train=False, download=True, transform=None)

        self.mean = self.train.mean
        self.std = self.train.std

    def train_dataloader(self):
        print('check=====')
        print('self.batch_size,', self.batch_size)
        loader = torch.utils.data.DataLoader(
            dataset=self.train,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            drop_last=True)
        return loader

    def val_dataloader(self):
        loader = torch.utils.data.DataLoader(
            dataset=self.val,
            batch_size=self.batch_size*10,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
            drop_last=False)
        return loader

