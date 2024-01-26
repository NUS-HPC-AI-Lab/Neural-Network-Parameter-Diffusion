import hydra.utils
import torch

from .base import BaseSystem
import torch.nn as nn
import pytorch_lightning as pl
import pdb


class VAESystem(BaseSystem):
    def __init__(self, config, **kwargs):
        super(VAESystem, self).__init__(config)
        print("VAESystem init")

    def forward(self, batch, **kwargs):
        output = self.model(batch)
        total_loss = output['loss']
        self.log('loss', total_loss.cpu().detach().mean().item(), on_epoch=True, prog_bar=True, logger=True)
        return total_loss

    def validation_step(self, batch, batch_idx, **kwargs):
        # todo
        good_param = batch[:10]
        input_accs = []
        for i, param in enumerate(good_param):
            acc, test_loss, output_list = self.task_func(param)
            input_accs.append(acc)
        print("input model accuracy:{}".format(input_accs))
        print("input model average accuracy:{}".format(sum(input_accs)/len(input_accs)))

        """
        # VAE generate
        """
        print('---------------------------------')
        print('Test the VAE model')
        ae_rec_accs = []
        vae_params = self.generate(10)
        for i, param in enumerate(vae_params):
            acc, test_loss, output_list = self.task_func(param)
            ae_rec_accs.append(acc)

        best_ae = max(ae_rec_accs)
        print(f'AE reconstruction models accuracy:{ae_rec_accs}')
        print(f'AE reconstruction models best accuracy:{best_ae}')
        print(f"AE reconstruction models average accuracy:{sum(ae_rec_accs)/len(ae_rec_accs)}")
        print('---------------------------------')
        self.log('vae_acc', best_ae)

    def generate(self, num):
        return self.model.sample(num)