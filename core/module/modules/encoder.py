## Adopted from https://github.com/rosinality/denoising-diffusion-pytorch with some minor changes.

import math
import pdb
import random

import torch
from torch import nn
import torch.nn.functional as F

# one dimension convolutional encoder
class ODEncoder(nn.Module):
    def __init__(self, in_dim_list, fold_rate, kernel_size, channel_list):
        super(ODEncoder, self).__init__()
        self.in_dim_list = in_dim_list
        self.fold_rate = fold_rate
        self.kernel_size = kernel_size

        # insert the first layer
        channel_list = [channel_list[0]] + channel_list
        self.channel_list = channel_list
        encoder = nn.ModuleList()
        layer_num = len(channel_list) - 1

        for i in range(layer_num):
            if_last = False
            if_start = i == 0
            if i == layer_num - 1:
                if_last = True
            layer = self.build_layer(in_dim_list[i], kernel_size, fold_rate,
                                     channel_list[i], channel_list[i+1], if_last, if_start)
            encoder.append(layer)
        self.encoder = encoder


    def build_layer(self, in_dim, kernel_size, fold_rate, input_channel, output_channel, last=False, if_start=False):
        # first: if is the first layer of encoder
        layer = nn.Sequential(
            nn.LeakyReLU() if not if_start else nn.Identity(),
            nn.InstanceNorm1d(in_dim),
            nn.Conv1d(input_channel if not if_start else 1, input_channel, kernel_size, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(in_dim),
            nn.Conv1d(input_channel, output_channel, kernel_size, stride=fold_rate, padding=0),
            nn.Tanh() if last else nn.Identity(),
        )
        return layer


    def forward(self, x, **kwargs):
        for i, module in enumerate(self.encoder):
            x = module(x)
        return x


class ODDecoder(nn.Module):
    def __init__(self, in_dim_list, fold_rate, kernel_size, channel_list):
        super(ODDecoder, self).__init__()
        self.in_dim_list = in_dim_list
        self.fold_rate = fold_rate
        self.kernel_size = kernel_size

        # insert the first layer
        channel_list = channel_list + [1]
        self.channel_list = channel_list
        decoder = nn.ModuleList()
        layer_num = len(channel_list) - 1

        for i in range(layer_num):
            if_last = False
            if i == layer_num - 1:
                if_last = True
            layer = self.build_layer(in_dim_list[i], kernel_size, fold_rate, channel_list[i], channel_list[i + 1],
                                     if_last)
            decoder.append(layer)
        self.decoder = decoder


    def build_layer(self, in_dim, kernel_size, fold_rate, input_channel, output_channel, last):
        layer = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(in_dim),
            nn.ConvTranspose1d(input_channel, input_channel, kernel_size, stride=fold_rate, padding=0),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(in_dim),
            nn.Conv1d(input_channel, output_channel, kernel_size, stride=1, padding=fold_rate if last else fold_rate - 1)
        )
        return layer

    def forward(self, x, **kwargs):
        for i, module in enumerate(self.decoder):
            x = module(x)
        return x

class ODEncoder2Decoder(nn.Module):
    def __init__(self, in_dim, kernel_size=3, fold_rate=3, input_noise_factor=0.001, latent_noise_factor=0.1, enc_channel_list=None, dec_channel_list=None):
        super(ODEncoder2Decoder, self).__init__()
        self.in_dim = in_dim
        self.fold_rate = fold_rate
        self.kernel_size = kernel_size
        self.input_noise_factor = input_noise_factor
        self.latent_noise_factor = latent_noise_factor
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

        self.real_input_dim = real_input_dim
        self.encoder = ODEncoder(enc_dim_list, fold_rate, kernel_size, enc_channel_list)
        self.decoder = ODDecoder(dec_dim_list, fold_rate, kernel_size, dec_channel_list)

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

    def add_noise(self, x, noise_factor):
        if not isinstance(noise_factor, float):
            assert len(noise_factor) == 2
            noise_factor = random.uniform(noise_factor[0], noise_factor[1])

        return torch.randn_like(x) * noise_factor + x * (1 - noise_factor)

    def forward(self, x, **kwargs):
        x = self.add_noise(x, self.input_noise_factor)
        x = self.encode(x)
        x = self.add_noise(x, self.latent_noise_factor)
        x = torch.clamp(x, -1, 1)
        x = self.decode(x)
        return x


class small(ODEncoder2Decoder):
    def __init__(self, in_dim, input_noise_factor, latent_noise_factor):
        fold_rate = 3
        kernel_size = 3
        enc_channel_list = [2, 2, 2, 2]
        dec_channel_list = [2, 64, 64, 8]
        super(small, self).__init__(in_dim, kernel_size, fold_rate, input_noise_factor, latent_noise_factor, enc_channel_list, dec_channel_list)

class medium(ODEncoder2Decoder):
    def __init__(self, in_dim, input_noise_factor, latent_noise_factor):
        fold_rate = 5
        kernel_size = 5
        enc_channel_list = [4, 4, 4, 4]
        dec_channel_list = [4, 256, 256, 8]
        super(medium, self).__init__(in_dim, kernel_size, fold_rate, input_noise_factor, latent_noise_factor, enc_channel_list, dec_channel_list)


if __name__ == '__main__':
    model = small(2048, 0.1, 0.1)