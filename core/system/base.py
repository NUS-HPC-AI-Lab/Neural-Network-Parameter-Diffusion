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

class BaseSystem(abc.ABC, pl.LightningModule):
    def __init__(self, cfg, task, **kwargs):
        super(BaseSystem, self).__init__()
        self.save_hyperparameters()
        self.task = task
        # when save  hyperparameters, the self.task will be ignored
        self.config = cfg
        self.automatic_optimization = False
        self.train_cfg = cfg.train
        self.model_cfg = cfg.model
        self.model = self.build_model()
        self.loss_func = self.build_loss_func()
        self.data_transform = self.build_data_transform()
        pass

    # def set_task(self, task):
    #     self.task = task
    #
    # def get_task(self):
    #     return self.task

    def build_data_transform(self):
        if 'data_transform' in self.model_cfg and self.model_cfg.data_transform is not None:
            return hydra.utils.instantiate(self.config.data_transform)
        else:
            return None

    def build_trainer(self):
        trainer = hydra.utils.instantiate(self.train_cfg.trainer)
        return trainer

    def task_func(self, input):
        return self.task.test_g_model(input)

    @staticmethod
    def system_training(system, datamodule, **kwargs):
        trainer = system.build_trainer()
        pdb.set_trace()
        trainer.fit(system, datamodule=datamodule)
        print("best model starting saving")

    # @staticmethod
    # def system_testing(system, **kwargs):
    #     trainer = system.build_trainer()
    #     data_module = system.get_task().get_param_data()
    #     trainer.test(system, datamodule=data_module)

    def training_step(self, batch, batch_idx, **kwargs):
        loss = self.forward(batch, **kwargs)
        self.optimizer.zero_grad()
        self.manual_backward(loss)
        self.optimizer.step()

        if hasattr(self, 'lr_scheduler'):
            self.lr_scheduler.step()

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
        raise
