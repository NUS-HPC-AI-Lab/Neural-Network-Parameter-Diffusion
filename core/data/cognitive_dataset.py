import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision
from .base import DataBase
import torch
import timm

from neurogym.wrappers import ScheduleEnvs
from neurogym.utils.scheduler import RandomSchedule
from neurogym.wrappers.block import MultiEnvs
from neurogym import Dataset
from neurogym.utils.plotting import *
from Mod_Cog.mod_cog_tasks import *


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
        self.dataset = getattr(self.cfg, 'dataset', 'cognitive')


    @property
    def data_cls(self):
        envs_arr = [go(), rtgo(), dlygo(), anti(), rtanti(), dlyanti(),
            dm1(), dm2(), ctxdm1(), ctxdm2(), multidm(), dlydm1(), dlydm2(),
            ctxdlydm1(), ctxdlydm2(), multidlydm(), dms(), dnms(), dmc(), dnmc()]

        task_name_arr = ['go', 'rtgo', 'dlygo', 'anti', 'rtanti', 'dlyanti',
                'dm1', 'dm2', 'ctxdm1', 'ctxdm2', 'multidm', 'dlydm1', 'dlydm2',
                'ctxdlydm1', 'ctxdlydm2', 'multidlydm', 'dms', 'dnms', 'dmc', 'dnmc']
    
        # TODO: Add multiple tasks' case
        try:
            index = task_name_arr.index(self.cfg.task)
            envs = [envs_arr[index]]

        except ValueError:
            envs = None
            print("Task name not found in the list.")

        return envs

# TASK
    @property
    def train_dataset(self):
        envs = self.data_cls
        schedule = RandomSchedule(len(envs))
        env = ScheduleEnvs(envs, schedule=schedule, env_input=True)
        dataset = Dataset(env, batch_size=self.cfg.batch_size, seq_len=self.cfg.seq_len)
        return dataset

    @property
    def val_dataset(self):
        envs = self.data_cls
        schedule = RandomSchedule(len(envs))
        env = ScheduleEnvs(envs, schedule=schedule, env_input=True)
        dataset = Dataset(env, batch_size=self.cfg.batch_size, seq_len=self.cfg.seq_len)
        return dataset
    
    @property
    def test_dataset(self):
        envs = self.data_cls
        schedule = RandomSchedule(len(envs))
        env = ScheduleEnvs(envs, schedule=schedule, env_input=True)
        dataset = Dataset(env, batch_size=self.cfg.batch_size, seq_len=self.cfg.seq_len)
        return dataset
