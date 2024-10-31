import torch
import torch.nn as nn
from torch.nn import functional as F
import math


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_dim, frequency_embedding_size=256, max_period=10000):
        super().__init__()
        assert frequency_embedding_size % 2 == 0
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_dim, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=True)
        )  # FIXME: this is too big! Why this is such necessary?
        half = frequency_embedding_size // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half) / half)
        self.register_buffer("freqs", freqs)

    def forward(self, t):
        args = t[:, None].float() * self.freqs
        t_freq = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        t_emb = self.mlp(t_freq)
        return t_emb


class ConditionalUNet(nn.Module):
    def __init__(self, layer_channels: list, model_dim: int, condition_dim: int, kernel_size: int):
        super().__init__()
        self.time_embedder = TimestepEmbedder(hidden_dim=model_dim)
        self.condi_embedder = nn.Linear(condition_dim, model_dim)
        # FIXME: condi_embedder is calculated for 1000 times as same, but it does not work in recurrent module, why?
        self.encoder_list = nn.ModuleList([])
        for i in range(len(layer_channels) // 2 + 1):
            self.encoder_list.append(nn.ModuleList([
                nn.Conv1d(layer_channels[i], layer_channels[i+1], kernel_size, 1, kernel_size // 2),
                nn.Sequential(nn.BatchNorm1d(layer_channels[i+1]), nn.ELU())
            ]))
        self.decoder_list = nn.ModuleList([])
        for i in range(len(layer_channels) // 2 + 1, len(layer_channels) - 1):
            self.decoder_list.append(nn.ModuleList([
                nn.Conv1d(layer_channels[i], layer_channels[i+1], kernel_size, 1, kernel_size // 2),
                nn.Sequential(nn.BatchNorm1d(layer_channels[i+1]), nn.ELU())
                    if layer_channels[i+1] != 1 else nn.Identity(),
            ]))

    def forward(self, x, t, c):
        x = x[:, None, :]
        t = self.time_embedder(t)[:, None, :]
        c = self.condi_embedder(c)[:, None, :]
        x_list = []
        for i, (module, activation) in enumerate(self.encoder_list):
            x = module((x + c) * t)
            x = activation(x)
            if i < len(self.encoder_list) - 2:
                x_list.append(x)
        for i, (module, activation) in enumerate(self.decoder_list):
            x = x + x_list[-i-1]
            x = module((x + c) * t)
            x = activation(x)
        return x[:, 0, :]


class OneDimCNN(nn.Module):
    def __init__(self, layer_channels: list, model_dim: int, kernel_size: int):
        super().__init__()
        self.time_embedder = TimestepEmbedder(hidden_dim=model_dim)
        self.encoder_list = nn.ModuleList([])
        for i in range(len(layer_channels) // 2 + 1):
            self.encoder_list.append(nn.ModuleList([
                nn.Conv1d(layer_channels[i], layer_channels[i+1], kernel_size, 1, kernel_size // 2),
                nn.Sequential(nn.BatchNorm1d(layer_channels[i+1]), nn.ELU())
            ]))
        self.decoder_list = nn.ModuleList([])
        for i in range(len(layer_channels) // 2 + 1, len(layer_channels) - 1):
            self.decoder_list.append(nn.ModuleList([
                nn.Conv1d(layer_channels[i], layer_channels[i+1], kernel_size, 1, kernel_size // 2),
                nn.Sequential(nn.BatchNorm1d(layer_channels[i+1]), nn.ELU())
                    if layer_channels[i+1] != 1 else nn.Identity(),
            ]))

    def forward(self, x, t, c=0.):
        x = x[:, None, :]
        t = self.time_embedder(t)[:, None, :]
        x_list = []
        for i, (module, activation) in enumerate(self.encoder_list):
            x = module(x + t)
            x = activation(x)
            if i < len(self.encoder_list) - 2:
                x_list.append(x)
        for i, (module, activation) in enumerate(self.decoder_list):
            x = x + x_list[-i-1]
            x = module(x + t)
            x = activation(x)
        return x[:, 0, :]