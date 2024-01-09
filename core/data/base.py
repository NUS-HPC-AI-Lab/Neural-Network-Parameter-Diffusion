import torch
import hydra
import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS


class DataBase(pl.LightningDataModule):
    def __init__(self, cfg, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.batch_size = getattr(cfg, 'batch_size', 32)
        self.num_workers = getattr(cfg, 'num_workers', 0)

    def prepare_data(self):
        pass

    @property
    def train_dataset(self):
        raise NotImplementedError

    @property
    def val_dataset(self):
        raise NotImplementedError

    @property
    def test_dataset(self):
        raise NotImplementedError


    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=True, drop_last=True, pin_memory=True, persistent_workers=True)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, persistent_workers=True)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, persistent_workers=True)

