import hydra.utils
import torch
import numpy as np
from .base import BaseSystem
import torch.nn as nn
import pytorch_lightning as pl
import pdb



class EncoderSystem(BaseSystem):
    def __init__(self, config, **kwargs):
        super(EncoderSystem, self).__init__(config)
        print("EncoderSystem init")
        self.save_hyperparameters()

    def forward(self, batch, **kwargs):
        output = self.model(batch)
        loss = self.loss_func(batch, output, **kwargs)
        # self.log('epoch', self.current_epoch)
        self.log('loss', loss.cpu().detach().mean().item(), on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx, **kwargs):
        # todo
        good_param = batch[:20]
        input_accs = []
        for i, param in enumerate(good_param):
            acc, test_loss, output_list = self.task_func(param)
            input_accs.append(acc)
        print("input model accuracy:{}".format(input_accs))

        """
        AE reconstruction parameters
        """
        print('---------------------------------')
        print('Test the AE model')
        ae_rec_accs = []
        ae_params = self.decode(self.encode((good_param)))
        for i, param in enumerate(ae_params):
            acc, test_loss, output_list = self.task_func(param)
            ae_rec_accs.append(acc)

        best_ae = max(ae_rec_accs)
        print(f'AE reconstruction models accuracy:{ae_rec_accs}')
        print(f'AE reconstruction models best accuracy:{best_ae}')
        print(f"AE reconstruction models average accuracy:{sum(ae_rec_accs) / len(ae_rec_accs)}")
        print(f"AE reconstruction models median accuracy:{np.median(ae_rec_accs)}")
        print('---------------------------------')

    def validation_step(self, batch, batch_idx, **kwargs):
        # todo
        good_param = batch[:10]
        input_accs = []
        for i, param in enumerate(good_param):
            acc, test_loss, output_list = self.task_func(param)
            input_accs.append(acc)
        print("input model accuracy:{}".format(input_accs))

        """
        AE reconstruction parameters
        """
        print('---------------------------------')
        print('Test the AE model')
        ae_rec_accs = []
        ae_params = self.decode(self.encode((good_param)))
        for i, param in enumerate(ae_params):
            acc, test_loss, output_list = self.task_func(param)
            ae_rec_accs.append(acc)

        best_ae = max(ae_rec_accs)
        print(f'AE reconstruction models accuracy:{ae_rec_accs}')
        print(f'AE reconstruction models best accuracy:{best_ae}')
        print('---------------------------------')
        self.log('ae_acc', best_ae)

    def encode(self, x, **kwargs):
        return self.model.encode(x)

    def decode(self, x, **kwargs):
        return self.model.decode(x)