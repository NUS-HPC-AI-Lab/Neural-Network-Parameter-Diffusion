
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
                backbone='AE_cnn',
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
        self.load_ae = False
        self.emstep = 'estep'
        self.automatic_optimization = False
        self.dataset = dataset
        self.quantized_inputs_mean=0
        self.quantized_inputs_std=1
        self.start = True
        self.target_layer = target_layer
        self.latent_train_epochs = latent_epoch
        self.num_params_data = int(param_num)#317706#594186#317706
        print(f"<<<<<<number of params is {self.num_params_data}>>>>>>>>")
        if self.num_params_data < 5000:
            self.AE_model = Latent_AE_cnn_small(self.num_params_data)
        
        elif self.num_params_data < 400000:
            self.AE_model = Latent_AE_cnn(self.num_params_data)
        
        else:
            print("self.num_params_data",self.num_params_data)
            # import pdb; pdb.set_trace()
            self.AE_model = Latent_AE_cnn_big(self.num_params_data,channel=int(channel))
        self.laten_dim = self.AE_model.Enc(torch.randn(1, 1, self.num_params_data)).shape
        print('===check===: latent feature dim is:', self.laten_dim)
        # init Unet:diffusionmodel的backbone
        if backbone == 'AE_cnn':
            self.UNet_model = AE_CNN_bottleneck(in_channel=1, in_dim=self.laten_dim[1]*self.laten_dim[2], dec=self.AE_model)
            self.UNet_ema   = AE_CNN_bottleneck(in_channel=1, in_dim=self.laten_dim[1]*self.laten_dim[2], dec=self.AE_model)
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
        self.evalloader = get_evaldata(self.dataset)
        self.testloader = get_testdata(self.dataset)
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
            # def training_step(self, batch, batch_nb):

        views = batch
        self.views = views
        img = self.AE_model.Enc(views).detach()
        # import pdb; pdb.set_trace()
        # img = views
        time = (torch.rand(img.shape[0]) * self.hparams.n_timestep).type(torch.int64).to(img.device)
        
        loss = self.diffusion.training_losses(self.UNet_model, img, time,).mean()

        accumulate(self.UNet_ema, self.UNet_model.module if isinstance(self.UNet_model, nn.DataParallel) else self.UNet_model, 0.9999)

        loss_dict = {
            'loss_diff': loss,
            'loss_all': loss,
            'epoch': self.current_epoch,
        }

        return loss_dict

    def train_latnet_model(self, batch, batch_nb):
        views = batch
        self.views = views
        img = views

        img_noised = img + torch.randn_like(img) * 0.001
        loss_ae = torch.nn.MSELoss()(self.AE_model(img_noised), img)
        
        loss_dict = {
            'loss_ae': loss_ae,
            'loss_diff': torch.tensor(100),
            'loss_all': loss_ae,
            'epoch': self.current_epoch,
        }
        return loss_dict

    def training_step(self, batch, batch_idx):
        optimizer_vqvae, optimizer_diff = self.optimizers()

        if not self.load_ae: #如果不load ae，则需要训ae
            
            if self.current_epoch < self.latent_train_epochs:
                loss_dict = self.train_latnet_model(batch, batch_idx)
                optimizer = optimizer_vqvae
            else:
                loss_dict = self.train_M_step(batch, batch_idx)
                optimizer = optimizer_diff
        else: #如果load ae，则直接开始训diffusion model
                loss_dict = self.train_M_step(batch, batch_idx)
                optimizer = optimizer_diff

        
        loss = loss_dict['loss_all']
        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()
        self.log_dict(loss_dict)
        
        
    # def augment_data_simple(self, lab):
    def augment_data_simple(self,generated_models_num=100):

        # shape  = (cond_input_val.shape[0], 3, self.hparams.data_resolution, self.hparams.data_resolution)
        shape  = (generated_models_num, 1, self.laten_dim[1] * self.laten_dim[2])

        self.UNet_ema.eval()
        sample = progressive_samples_fn_simple(
            self.UNet_ema, 
            self.diffusion, 
            shape, 
            device='cuda',
            # cond = cond_input_val,
            include_x0_pred_freq=50,
            )
        sample['samples'] = self.AE_model.Dec(sample['samples'].reshape(
            generated_models_num, self.laten_dim[1], self.laten_dim[2]))

        
        return sample['samples']# , sample['progressive_samples']
    def validation_step(self, batch, batch_idx): #一个validation epoch里面会按照batch切分为许多step
        #self.imgs = []  self.samples
        #self.views = torch.cat((batch[:,-5120:-3072],batch[:,-2048:]),1)
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
        pl.seed_everything(42)
        """
        good parameters
        """
        good_param = self.views[:10]
        input_accs = []
        for param in good_param:
            acc = test_generated_partial(self,param,self.evalloader)
            input_accs.append(acc)
        print(f'Input models accuracy:{input_accs}')  
        mean_input = np.mean(np.array(input_accs))

        top1_params = top_acc_params(self,input_accs,good_param,1)
        best_acc_input = test_generated_partial(self,top1_params,self.testloader)
        print(f'Average Input models accuracy:{mean_input}')
        print(f'Best Input models accuracy:{best_acc_input}')  
        """
        AE reconstruction parameters
        """
        print('---------------------------------')
        print('Test the AE model')
        ae_rec_accs = []
        for param in good_param:
            acc = test_generated_partial(self, self.AE_model(param.reshape(1,1,self.num_params_data)),self.evalloader)
            ae_rec_accs.append(acc)
        mean_ae = np.mean(np.array(ae_rec_accs))
        top1_params = top_acc_params(self,ae_rec_accs,good_param,1)
        best_ae = test_generated_partial(self, self.AE_model(top1_params.reshape(1,1,self.num_params_data)),self.testloader)
        print(f'AE reconstruction models accuracy:{ae_rec_accs}')       
        print(f'AE reconstruction models best accuracy:{best_ae}')       
        print('---------------------------------')
        
        

        """
        Diffusion generated parameters
        """
        ## 调整逻辑为：当latent ae训练完毕后，才做diffusion生成模型的测试
        if self.current_epoch>self.latent_train_epochs:
            accs = []
            best_params = []
            all_params = []
            gen_gen_dis_all =[]
            good_gen_dis_all = []
            for batch in range(10):
                self.samples = self.augment_data_simple(generated_models_num=10)
                for i in range(self.samples.shape[0]):
                    param = self.samples[i]
                    acc = test_generated_partial(self, param.reshape(1,1,self.num_params_data),self.evalloader)
                    accs.append(acc)
                gen_gen_dis_all.append(self.distance_model_list(self.samples, self.samples).cpu())
                good_gen_dis_all.append(self.distance_model_list(self.samples, good_param).cpu())
                all_params.append(self.samples)
                del self.samples
            all_params = torch.cat(all_params,0)
            top10_params = top_acc_params(self,accs,all_params,10)    
            top1_params = top_acc_params(self,accs,all_params,1)
            best_acc_generated = test_generated_partial(self,top1_params,self.testloader)
            ensem_acc = test_ensem_partial(self, top10_params,self.testloader)
            mean = np.mean(np.array(accs))
            worst = min(np.array(accs))
            median = np.median(np.array(accs))

            good_good_dis = self.distance_model_list(good_param, good_param)
            gen_gen_dis = np.mean(np.array(gen_gen_dis_all))
            good_gen_dis = np.mean(np.array(good_gen_dis_all))

            print(f'Generated models test accuracy:{accs[:20]}')
            print(f'Average generated models test accuracy:{mean}')
            print(f'Best generated accuracy is {best_acc_generated}')
            print(f'Ensemble generated accuracy is {ensem_acc}')
            print(f'good_good_dis:{good_good_dis}, gen_gen_dis:{gen_gen_dis}, good_gen_dis:{good_gen_dis}')
        else:
            accs = [-1]
            ensem_acc = -1
            mean = -1
            best_acc_generated = -1
            worst = -1
            median = -1
            good_good_dis = -1
            gen_gen_dis = -1
            good_gen_dis = -1


       
        self.logger.experiment.log(
            {
                'Average Test Acc of generated model': mean,
                'Average Test Acc of AE model': mean_ae,
                # 'Average Test Acc of input model': mean_input,
                'Best Test Acc of input model': best_acc_input,

                'Best Acc of AE model': best_ae,
                'Best Acc of generated model': best_acc_generated,
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
        params_vqvae = self.AE_model.parameters()
        params_diff = self.UNet_model.parameters()
        if self.hparams.optimazer == 'adam':
            pass
            # optimizer_vqvae = torch.optim.Adam(params_vqvae, lr=self.hparams.vq_lr, weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimazer == 'adamw':
            optimizer_vqvae = torch.optim.AdamW(params_vqvae, lr=self.hparams.vq_lr, weight_decay=self.hparams.weight_decay)
            optimizer_diff = torch.optim.AdamW(params_diff, lr=self.hparams.diff_lr, weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimazer == 'sgd':
            # optimizer_vqvae = torch.optim.SGD(params_vqvae, lr=self.hparams.vq_lr, weight_decay=self.hparams.weight_decay)
            optimizer_diff = torch.optim.SGD(params_diff, lr=self.hparams.diff_lr, weight_decay=self.hparams.weight_decay)

        return optimizer_vqvae, optimizer_diff

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
    
if __name__ == '__main__':
    #import os;os.environ["WANDB_MODE"] = "offline" 
    main()