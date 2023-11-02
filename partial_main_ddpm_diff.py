
"""
本文件执行 不带latent ae的 diffusion model
提供两种backbone: ae_fc,ae_cnn
"""


import os
import lightning as pl
import time
from torch import nn
import pdb
import torchvision.transforms as transforms
import numpy as np
from typing import List, Callable, Union, Any, TypeVar, Tuple
Tensor = TypeVar('torch.tensor')
import torchvision
import torchvision.utils as vutils
from data_utils.myDataModule import ParametersDataModule_partial
from lightning.pytorch.cli import LightningCLI
import torch
import wandb
# from data_utils.models import *
from utils import get_net,get_good, get_testdata, test, test_ensem, test_generated_partial, test_ensem_partial, save_best10,get_evaldata,top_acc_params
from lib.model import *
from lib.model_Latent import *
from lib.diffusion import GaussianDiffusion, make_beta_schedule
from models import *
torch.set_num_threads(2)

aug_data_list = []
aug_data_index_list = []


def accumulate(model1, model2, decay=0.9999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)

def progressive_samples_fn(model, diffusion, shape, device, cond, include_x0_pred_freq=50):
    samples, progressive_samples = diffusion.p_sample_loop_progressive(
        model=model,
        shape=shape,
        noise_fn=torch.randn,
        device=device,
        include_x0_pred_freq=include_x0_pred_freq,
        # cond=cond,
    )
    # return {'samples': (samples + 1)/2, 'progressive_samples': (progressive_samples + 1)/2}
    return {'samples': samples, 'progressive_samples': progressive_samples}

# def progressive_samples_fn_simple(model, diffusion, shape, device, cond, include_x0_pred_freq=50):
def progressive_samples_fn_simple(model, diffusion, shape, device, include_x0_pred_freq=50):

    samples, history = diffusion.p_sample_loop_progressive_simple(
        model=model,
        shape=shape,
        noise_fn=torch.randn,
        device=device,
        include_x0_pred_freq=include_x0_pred_freq,
        # cond=cond,
    )
    return {'samples': samples}

import random
class RandomApply(nn.Module):
    def __init__(self, fn: Callable, p: float):
        super().__init__()
        self.fn = fn
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        return x if random.random() > self.p else self.fn(x)

class AutoAug_AllinOne(pl.LightningModule):

    def __init__(self, 
                v_input=100, 
                v_latent=2, 
                augNearRate=100000, 
                sigmaP=1.0, 
                base_momentum=0.99,
                batch_size=256,
                arch='resnet18',
                hidden_dim=4096,
                proj_dim=256,
                optimazer='adamw',
                warmup_epochs=10,
                max_epochs=2000,
                dmtlosstype='latent',
                vq_lr=1e-5,
                linear_loss_weight='1.0',
                loss_recons_weight='1.0',
                vq_loss_weight='1.0',
                weight_decay=1.0e-6,
                #  -----------
                diff_lr=2e-4, #1e-4
                #  -----------
                schedule_type='linear',
                schedule_start=1e-4,
                schedule_end=2e-2,
                n_timestep=1000,#1000,
                #  -----------
                diff_mean_type='eps',
                diff_var_type='fixedlarge',
                diff_loss_type='mse',
                backbone='AE_cnn_ori',
                num_model=1,
                param_num=277,
                network=None,
                dataset='mnist',
                channel=4,
                latent_epoch=2000,
                target_layer='conv1',
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.emstep = 'estep'
        self.automatic_optimization = False
        self.dataset = dataset
        self.quantized_inputs_mean=0
        self.quantized_inputs_std=1
        self.start = True
        self.target_layer = target_layer
        self.num_params_data = int(param_num)#317706#594186#317706
        print(f"<<<<<<number of params is {self.num_params_data}>>>>>>>>")

        # init Unet:diffusionmodel的backbone
        if backbone == 'AE_cnn_ori':
            print("======use convAE as diffusion backbone======")
            self.UNet_model = AE_CNN_bottleneck_original(in_dim=self.num_params_data)
            self.UNet_ema   = AE_CNN_bottleneck_original(in_dim=self.num_params_data)
        elif backbone == 'AE_cnn_deep':
            print("======use convAE as diffusion backbone======")
            self.UNet_model = AE_CNN_bottleneck_deep(in_dim=self.num_params_data)
            self.UNet_ema   = AE_CNN_bottleneck_deep(in_dim=self.num_params_data)
        # elif backbone == 'AE_fc':
        #     print("======use fcAE as diffusion backbone======")
        #     self.UNet_model = AE(in_dim=param_num)
        #     self.UNet_ema   = AE(in_dim=param_num)

        self.betas = make_beta_schedule(schedule=schedule_type,
                                        start=schedule_start,
                                        end=schedule_end,
                                        n_timestep=n_timestep)
        # 可以理解为loss function 
        self.diffusion = GaussianDiffusion(betas=self.betas,
                                           model_mean_type=diff_mean_type,
                                           model_var_type=diff_var_type,
                                           loss_type=diff_loss_type)   


        self.acc1_list = []
        self.train_feats_list = []
        self.train_labels_list = []
        self.train_recons_list = []
        
        self.log_dict({
            'loss_diff': torch.tensor(1.0),
            'epoch': self.current_epoch,
        })

        # for validation
        self.generated = []
        self.real = []
        # for test for generated model
        self.testloader = get_testdata(self.dataset,network=network)
        self.evalloader = get_evaldata(self.dataset,len_s=2048,network=network)
        
        net = get_net(self.dataset, self.num_params_data,network)
        self.net = net
    def distance_model_list(self, m1, m2):
        
        # import pdb; pdb.set_trace()
        dis_list = []
        for i in range(m1.shape[0]):
            for j in range(m2.shape[0]):
                dis_list.append(torch.norm(m1[i] - m2[j], p=2).mean())
        
        return torch.stack(dis_list).mean()

    def collect_params(self, models, exclude_bias_and_bn=True):
        param_list = []
        for model in models:
            for name, param in model.named_parameters():
                if exclude_bias_and_bn and any(
                    s in name for s in ['bn', 'downsample.1', 'bias']):
                    param_dict = {
                        'params': param,
                        'weight_decay': 0.,
                        'lars_exclude': True}
                    # NOTE: with the current pytorch lightning bolts
                    # implementation it is not possible to exclude 
                    # parameters from the LARS adaptation
                else:
                    param_dict = {'params': param}
                param_list.append(param_dict)
        return param_list

    def emModelchange(self,):
        inter = 200
        s = self.current_epoch//inter
        if s % 2 == 1 and self.emstep == 'estep':
            self.emstep='mstep'
        elif s % 2 == 0 and self.emstep == 'mstep':
            self.emstep='estep'

    def train_M_step(self, batch, batch_nb):
        views = batch
        self.views = views
        img = views
        time = (torch.rand(img.shape[0]) * self.hparams.n_timestep).type(torch.int64).to(img.device)
        loss = self.diffusion.training_losses(self.UNet_model, views, time,).mean()

        accumulate(self.UNet_ema, self.UNet_model.module if isinstance(self.UNet_model, nn.DataParallel) else self.UNet_model, 0.9999)

        loss_dict = {
            'loss_diff': loss,
            'loss_all': loss,
            'epoch': self.current_epoch,
        }

        return loss_dict

    def on_train_epoch_end(self):
        lr_scheduler = self.lr_schedulers()
        lr_scheduler.step()
    def training_step(self, batch, batch_idx):
        print("train_bath",batch.shape)
        optimizer_diff = self.optimizers()
        # self.vqmodel.eval()
        loss_dict = self.train_M_step(batch, batch_idx)
        # wandb.log(loss_dict)
        loss = loss_dict['loss_all']
        
        optimizer_diff.zero_grad()
        self.manual_backward(loss)
        optimizer_diff.step()
        self.log_dict(loss_dict)
        self.log_dict({
            'lr':self.lr_schedulers().get_lr()[0]
        })
        
        
    # def augment_data_simple(self, lab):
    def augment_data_simple(self):

        # shape  = (cond_input_val.shape[0], 3, self.hparams.data_resolution, self.hparams.data_resolution)
        generated_models_num = 100
        shape  = (generated_models_num, 1, self.num_params_data)

        self.UNet_ema.eval()
        sample = progressive_samples_fn_simple(
            self.UNet_ema, 
            self.diffusion, 
            shape, 
            device='cuda',
            # cond = cond_input_val,
            include_x0_pred_freq=50,
            )

        
        return sample['samples']# , sample['progressive_samples']
    def validation_step(self, batch, batch_idx): #一个validation epoch里面会按照batch切分为许多step
        self.views = batch
        # import pdb; pdb.set_trace()

        if self.start:
            self.log_dict({
                'loss_diff': torch.tensor(1.0),
                'epoch': self.current_epoch,
            })
            self.start = False




    def on_validation_epoch_end(self) -> None: 
        #每个validation epoch调用一次  validation epoch出现在指定train epochs之后
        print('validation_epoch_end start')
        
        pl.seed_everything(42)
        self.samples = self.augment_data_simple()
        # print(f'=====check====={self.samples.shape}') torch.[10,1,5066]

        good_param = self.views[:10]
        input_accs = []
        for param in good_param:
            # import pdb; pdb.set_trace()
            
            acc = test_generated_partial(self,param,self.testloader)
            input_accs.append(acc)
        mean_input = np.mean(np.array(input_accs))
        best_acc_input = np.max(np.array(input_accs))
        

        print(f'Input models accuracy:{input_accs}')       
        print(f'Input models best accuracy:{best_acc_input}')       
        print('---------------------------------') 
        accs = []
        for i in range(self.samples.shape[0]):
            param = self.samples[i]
            acc = test_generated_partial(self, param.reshape(1,1,self.num_params_data),self.evalloader)
            accs.append(acc)
        best_params = save_best10(self,accs, self.samples)
        
        all_acc = torch.tensor(accs)
        better_acc_values,better_acc_indices = torch.topk(all_acc, 1, dim=0, largest=True)
        better_param = self.samples[better_acc_indices]
        better_acc_generated = test_generated_partial(self,better_param,self.testloader)

        ensem_acc = test_ensem_partial(self, best_params,self.testloader)
        mean = np.mean(np.array(accs))
        best = max(np.array(accs))
        worst = min(np.array(accs))
        median = np.median(np.array(accs))


        print(f'Generated models test accuracy:{accs}')
        print(f'Average generated models test accuracy:{mean}')
        print(f'Best generated accuracy is {best}')
        
        good_good_dis = self.distance_model_list(good_param, good_param)
        gen_gen_dis = self.distance_model_list(self.samples, self.samples)
        good_gen_dis = self.distance_model_list(self.samples, good_param)
        del self.samples
        print(f'good_good_dis:{good_good_dis}, gen_gen_dis:{gen_gen_dis}, good_gen_dis:{good_gen_dis}')
        self.logger.experiment.log(
            {
                'Average Test Acc of generated model': mean,
                'Average Test Acc of input model': mean_input,
                'Best Test Acc of input model': best_acc_input,
                'Best Acc of generated model': better_acc_generated,
                'Worst Acc of generated model': worst,
                'Median Acc of generated model': median,
                'Ensemble Acc':ensem_acc,
                'epoch': self.current_epoch,
                'good_good_dis':good_good_dis,
                'gen_gen_dis':gen_gen_dis, 
                'good_gen_dis':good_gen_dis
             }
            )



    def configure_optimizers(self):
        # params_vqvae = self.vqmodel.parameters()
        params_diff = self.UNet_model.parameters()
        if self.hparams.optimazer == 'adam':
            pass
            # optimizer_vqvae = torch.optim.Adam(params_vqvae, lr=self.hparams.vq_lr, weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimazer == 'adamw':
            # optimizer_vqvae = torch.optim.AdamW(params_vqvae, lr=self.hparams.vq_lr, weight_decay=self.hparams.weight_decay)
            optimizer_diff = torch.optim.AdamW(params_diff, lr=self.hparams.diff_lr, weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimazer == 'sgd':
            # optimizer_vqvae = torch.optim.SGD(params_vqvae, lr=self.hparams.vq_lr, weight_decay=self.hparams.weight_decay)
            optimizer_diff = torch.optim.SGD(params_diff, lr=self.hparams.diff_lr, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_diff, milestones=[800, 2000, 10000], gamma=0.1)
        return {"optimizer":optimizer_diff,"lr_scheduler":scheduler}

class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments("trainer.max_epochs", "model.max_epochs")
        parser.link_arguments("data.batch_size", "model.batch_size")
        parser.link_arguments("data.num_model", "model.num_model")

def main():

    print('==========================')
    print('Neural Networks Diffusion!')
    print('==========================')

    #lighting是pytorch封装  CLI是命令行控制工具
    cli = MyLightningCLI(
        AutoAug_AllinOne, 
        ParametersDataModule_partial,
        save_config_callback=None
        )
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)
    
if __name__ == '__main__':
    #import os;os.environ["WANDB_MODE"] = "offline"
    main()
