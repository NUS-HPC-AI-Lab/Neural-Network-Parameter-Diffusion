import hydra.utils
import torch
import pytorch_lightning as pl
import sys
import os
import datetime
from core.tasks.classification import CFTask
from core.system import *
import torch
import torch.distributed as dist

tasks = {
    'classification': CFTask,
}

system = {
    'encoder': EncoderSystem,
    'ddpm': DDPM,
}

class Runner(object):
    def __init__(self, cfg, **kwargs):
        self.cfg = cfg

        for k, v in kwargs.items():
            print(k, v)

        # set seed
        self.set_seed(cfg.seed)
        # set device
        self.set_device(cfg.device)

        # set process title
        self.set_processtitle()

    def set_processtitle(self):
        # set process title
        import setproctitle
        setproctitle.setproctitle(self.cfg.process_title)

    def train_generation(self, **kwargs):
        cfg = self.cfg
        # build task
        task_cls = tasks[cfg.task.name]
        self.task = task_cls(cfg.task, **kwargs)
        test_func = self.task.test_g_model

        # build system
        self.system_cls = system[cfg.system.name]
        self.system = self.system_cls(cfg.system, test_func, **kwargs)

        # running
        self.output_dir = cfg.output_dir
        datamodule = self.task.get_param_data()
        self.system_cls.system_training(self.system, datamodule)

        return {}

    def train_task_for_data(self, **kwargs):
        task_cls = tasks[self.cfg.task.name]
        self.task = task_cls(self.cfg.task, **kwargs)

        task_result = self.task.train_for_data()
        return task_result

    def set_seed(self, seed):
        pl.seed_everything(seed)

    def set_device(self, device_config):
        # set the global cuda device
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_config.cuda_visible_devices)
        torch.cuda.set_device(device_config.cuda)
        torch.set_float32_matmul_precision('medium')
        # warnings.filterwarnings("always")



