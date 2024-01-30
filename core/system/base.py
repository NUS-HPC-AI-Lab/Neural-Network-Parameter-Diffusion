import pdb

import pytorch_lightning as pl
import abc
import hydra
import torch.optim.lr_scheduler
import warnings
from typing import Optional, Union, List, Dict, Any, Sequence
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
import types
from core.tasks import tasks

class BaseSystem(pl.LightningModule, abc.ABC):
    def __init__(self, cfg):
        super(BaseSystem, self).__init__()
        # when save  hyperparameters, the self.task will be ignored
        self.save_hyperparameters()
        task_cfg =  cfg.task
        self.automatic_optimization = False
        self.config = cfg.system
        self.train_cfg = self.config.train
        self.model_cfg = self.config.model
        self.model = self.build_model()
        self.loss_func = self.build_loss_func()
        self.data_transform = self.build_data_transform()
        self.build_task(task_cfg)

    def build_task(self, task_cfg, **kwargs):
        self.task = tasks[task_cfg.name](task_cfg)

    def get_task(self):
        return self.task

    def build_data_transform(self):
        if 'data_transform' in self.config and self.config.data_transform is not None:
            return hydra.utils.instantiate(self.config.data_transform)
        else:
            return None

    def build_trainer(self):
        trainer = hydra.utils.instantiate(self.train_cfg.trainer)
        pdb.set_trace()
        return trainer

    def task_func(self, input):
        return self.task.test_g_model(input)

    def training_step(self, batch, batch_idx, **kwargs):
        optimizer = self.optimizers()
        loss = self.forward(batch, **kwargs)
        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()

        if hasattr(self, 'lr_scheduler'):
            self.lr_scheduler.step()

        return {'loss': loss}

    def build_model(self, **kwargs):
        model = hydra.utils.instantiate(self.model_cfg.arch)
        return model

    def build_loss_func(self):
        if 'loss_func' in self.train_cfg:
            loss_func = hydra.utils.instantiate(self.train_cfg.loss_func)
            return loss_func
        else:
            warnings.warn("No loss function is specified, using default loss function")


    def configure_optimizers(self, **kwargs):
        params = self.model.parameters()
        self.optimizer = hydra.utils.instantiate(self.train_cfg.optimizer, params)

        if 'lr_scheduler' in self.train_cfg and self.train_cfg.lr_scheduler is not None:
            self.lr_scheduler = hydra.utils.instantiate(self.train_cfg.lr_scheduler)

        return self.optimizer

    def validation_step(self, batch, batch_idx, **kwargs):
        # TODO using task layer
        pass

    @abc.abstractmethod
    def forward(self, x, **kwargs):
        raise NotImplementedError