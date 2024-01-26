import pdb

import torch

from .encoder import ODEncoder2Decoder
from torch import Tensor
import torch.nn as nn

## Adopted from https://github.com/rosinality/denoising-diffusion-pytorch with some minor changes.

import math
import random

import torch
from torch import nn
import torch.nn.functional as F
from .encoder import ODEncoder, ODDecoder


class ODVAE(nn.Module):
    def __init__(self, in_dim, latent_dim, kld_weight=0.5, kernel_size=3, fold_rate=3,
                 enc_channel_list=None, dec_channel_list=None):
        super(ODVAE, self).__init__()
        self.in_dim = in_dim
        self.fold_rate = fold_rate
        self.kernel_size = kernel_size
        self.kld_weight = kld_weight
        # default augment for debug
        if enc_channel_list is None:
            enc_channel_list = [2, 2, 2, 2]
        if dec_channel_list is None:
            dec_channel_list = [2, 64, 64, 8]

        enc_dim_list = []
        dec_dim_list = []

        enc_layer_num = len(enc_channel_list)
        real_input_dim = (
                int(in_dim / self.fold_rate ** enc_layer_num + 1) * self.fold_rate ** enc_layer_num
        )

        for i in range(len(enc_channel_list)):
            dim = real_input_dim // fold_rate ** i
            enc_dim_list.append(dim)

        for i in range(len(dec_channel_list)):
            dim = real_input_dim // fold_rate ** (4 - i)
            dec_dim_list.append(dim)

        self.enc_dim_list = enc_dim_list
        self.dec_dim_list = dec_dim_list
        self.enc_channel_list =  enc_channel_list
        self.dec_channel_list =  dec_channel_list

        self.real_input_dim = real_input_dim
        self.encoder = ODEncoder(enc_dim_list, fold_rate, kernel_size, enc_channel_list)
        self.decoder = ODDecoder(dec_dim_list, fold_rate, kernel_size, dec_channel_list)

        self.fc_mu =  nn.Linear(latent_dim, latent_dim)
        self.fc_var = nn.Linear(latent_dim, latent_dim)

    def encode(self, x, **kwargs):
        x = self.adjust_input(x)
        return self.encoder(x, **kwargs)

    def decode(self, x, **kwargs):
        decoded = self.decoder(x, **kwargs)
        return self.adjust_output(decoded)

    def adjust_output(self, output):
        return output[:, :, :self.in_dim].squeeze(1)

    def adjust_input(self, input):
        input_shape = input.shape
        if len(input.size()) == 2:
            input = input.view(input.size(0), 1, -1)

        input = torch.cat(
            [
                input,
                torch.zeros(input.shape[0], 1, (self.real_input_dim - self.in_dim)).to(
                    input.device
                ),
            ],
            dim=2,
        )
        return input

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def add_noise(self, x, noise_factor):
        if not isinstance(noise_factor, float):
            assert len(noise_factor) == 2
            noise_factor = random.uniform(noise_factor[0], noise_factor[1])

        return torch.randn_like(x) * noise_factor + (1 - noise_factor) * x

    def forward(self, input, **kwargs):
        x = self.encode(input)
        x_flatten = x.view(x.size(0), -1)
        mu = self.fc_mu(x_flatten)
        log_var = self.fc_var(x_flatten)
        z = self.reparameterize(mu, log_var)
        z = z.view(x.shape)
        z = torch.clamp(z, -1, 1)
        output = self.decode(z)

        # calculate loss
        recons = output
        recons_loss = F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        loss = recons_loss + self.kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss.detach(), 'KLD': -kld_loss.detach()}

    def sample(self, num_samples):
        z = torch.randn(num_samples, self.dec_channel_list[0], self.dec_dim_list[0]).cuda()
        samples = self.decode(z)
        return samples


class small(ODVAE):
    def __init__(self, in_dim, latent_dim, kld_weight):
        fold_rate = 3
        kernel_size = 3
        enc_channel_list = [2, 2, 2, 2]
        dec_channel_list = [2, 64, 64, 8]
        super(small, self).__init__(in_dim, latent_dim, kld_weight, kernel_size, fold_rate, enc_channel_list, dec_channel_list)

class medium(ODVAE):
    def __init__(self, in_dim, latent_dim, kld_weight):
        fold_rate = 5
        kernel_size = 5
        enc_channel_list = [4, 4, 4, 4]
        dec_channel_list = [4, 256, 256, 8]
        super(medium, self).__init__(in_dim, latent_dim, kld_weight, kernel_size, fold_rate, enc_channel_list, dec_channel_list)


if __name__ == '__main__':
    model = small(2048, 0.005)