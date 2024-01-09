## Adopted from https://github.com/rosinality/denoising-diffusion-pytorch with some minor changes.

import math

import torch
from torch import nn
import torch.nn.functional as F


def swish(input):
    return input * torch.sigmoid(input)


@torch.no_grad()
def variance_scaling_init_(tensor, scale=1, mode="fan_avg", distribution="uniform"):
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)

    if mode == "fan_in":
        scale /= fan_in

    elif mode == "fan_out":
        scale /= fan_out

    else:
        scale /= (fan_in + fan_out) / 2

    if distribution == "normal":
        std = math.sqrt(scale)

        return tensor.normal_(0, std)

    else:
        bound = math.sqrt(3 * scale)

        return tensor.uniform_(-bound, bound)


def conv2d(
        in_channel,
        out_channel,
        kernel_size,
        stride=1,
        padding=0,
        bias=True,
        scale=1,
        mode="fan_avg",
):
    conv = nn.Conv2d(
        in_channel, out_channel, kernel_size, stride=stride, padding=padding, bias=bias
    )

    variance_scaling_init_(conv.weight, scale, mode=mode)

    if bias:
        nn.init.zeros_(conv.bias)

    return conv


def linear(in_channel, out_channel, scale=1, mode="fan_avg"):
    lin = nn.Linear(in_channel, out_channel)

    variance_scaling_init_(lin.weight, scale, mode=mode)
    nn.init.zeros_(lin.bias)

    return lin


class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return swish(input)


class Upsample(nn.Sequential):
    def __init__(self, channel):
        layers = [
            nn.Upsample(scale_factor=2, mode="nearest"),
            conv2d(channel, channel, 3, padding=1),
        ]

        super().__init__(*layers)


class Downsample(nn.Sequential):
    def __init__(self, channel):
        layers = [conv2d(channel, channel, 3, stride=2, padding=1)]

        super().__init__(*layers)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, time_dim, dropout):
        super().__init__()

        self.norm1 = nn.GroupNorm(32, in_channel)
        self.activation1 = Swish()
        self.conv1 = conv2d(in_channel, out_channel, 3, padding=1)

        self.time = nn.Sequential(Swish(), linear(time_dim, out_channel))

        self.norm2 = nn.GroupNorm(32, out_channel)
        self.activation2 = Swish()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = conv2d(out_channel, out_channel, 3, padding=1, scale=1e-10)

        if in_channel != out_channel:
            self.skip = conv2d(in_channel, out_channel, 1)

        else:
            self.skip = None

    def forward(self, input, time):
        batch = input.shape[0]

        out = self.conv1(self.activation1(self.norm1(input)))

        out = out + self.time(time).view(batch, -1, 1, 1)

        out = self.conv2(self.dropout(self.activation2(self.norm2(out))))

        if self.skip is not None:
            input = self.skip(input)

        return out + input


class SelfAttention(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        self.norm = nn.GroupNorm(32, in_channel)
        self.qkv = conv2d(in_channel, in_channel * 4, 1)
        self.out = conv2d(in_channel, in_channel, 1, scale=1e-10)

    def forward(self, input):
        batch, channel, height, width = input.shape

        norm = self.norm(input)
        qkv = self.qkv(norm)
        query, key, value = qkv.chunk(3, dim=1)

        attn = torch.einsum("nchw, ncyx -> nhwyx", query, key).contiguous() / math.sqrt(
            channel
        )
        attn = attn.view(batch, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, height, width, height, width)

        out = torch.einsum("nhwyx, ncyx -> nchw", attn, input).contiguous()
        out = self.out(out)

        return out + input


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.dim = dim
        # half_dim      = self.dim // 2
        half_dim = self.dim
        self.inv_freq = torch.exp(
            torch.arange(half_dim, dtype=torch.float32)
            * (-math.log(10000) / (half_dim - 1))
        )

    def forward(self, input):
        import pdb

        pdb.set_trace()
        shape = input.shape
        input = input.view(-1).to(torch.float32)
        sinusoid_in = torch.ger(input, self.inv_freq.to(input.device))
        pos_emb = torch.cat([sinusoid_in.sin(), sinusoid_in.cos()], dim=-1)
        pos_emb = pos_emb.view(*shape, self.dim)

        return pos_emb


class ResBlockWithAttention(nn.Module):
    def __init__(self, in_channel, out_channel, time_dim, dropout, use_attention=False):
        super().__init__()

        self.resblocks = ResBlock(in_channel, out_channel, time_dim, dropout)

        if use_attention:
            self.attention = SelfAttention(out_channel)

        else:
            self.attention = None

    def forward(self, input, time):
        out = self.resblocks(input, time)

        if self.attention is not None:
            out = self.attention(out)

        return out


def spatial_fold(input, fold):
    if fold == 1:
        return input

    batch, channel, height, width = input.shape
    h_fold = height // fold
    w_fold = width // fold

    return (
        input.view(batch, channel, h_fold, fold, w_fold, fold)
            .permute(0, 1, 3, 5, 2, 4)
            .reshape(batch, -1, h_fold, w_fold)
    )


def spatial_unfold(input, unfold):
    if unfold == 1:
        return input

    batch, channel, height, width = input.shape
    h_unfold = height * unfold
    w_unfold = width * unfold

    return (
        input.view(batch, -1, unfold, unfold, height, width)
            .permute(0, 1, 4, 2, 5, 3)
            .reshape(batch, -1, h_unfold, w_unfold)
    )


class UNet(nn.Module):
    def __init__(
            self,
            in_channel,
            channel,
            channel_multiplier,
            n_res_blocks,
            attn_strides,
            dropout=0,
            fold=1,
            num_classes=10,  # for cifar10
    ):
        super().__init__()

        self.fold = fold

        time_dim = channel * 4

        n_block = len(channel_multiplier)

        self.time = nn.Sequential(
            TimeEmbedding(channel),
            linear(channel, time_dim),
            Swish(),
            linear(time_dim, time_dim),
        )

        down_layers = [conv2d(in_channel * (fold ** 2), channel, 3, padding=1)]
        feat_channels = [channel]
        in_channel = channel
        for i in range(n_block):
            for _ in range(n_res_blocks):
                channel_mult = channel * channel_multiplier[i]

                down_layers.append(
                    ResBlockWithAttention(
                        in_channel,
                        channel_mult,
                        time_dim,
                        dropout,
                        use_attention=2 ** i in attn_strides,
                    )
                )

                feat_channels.append(channel_mult)
                in_channel = channel_mult

            if i != n_block - 1:
                down_layers.append(Downsample(in_channel))
                feat_channels.append(in_channel)

        self.down = nn.ModuleList(down_layers)

        self.mid = nn.ModuleList(
            [
                ResBlockWithAttention(
                    in_channel,
                    in_channel,
                    time_dim,
                    dropout=dropout,
                    use_attention=True,
                ),
                ResBlockWithAttention(
                    in_channel, in_channel, time_dim, dropout=dropout
                ),
            ]
        )

        up_layers = []
        for i in reversed(range(n_block)):
            for _ in range(n_res_blocks + 1):
                channel_mult = channel * channel_multiplier[i]

                up_layers.append(
                    ResBlockWithAttention(
                        in_channel + feat_channels.pop(),
                        channel_mult,
                        time_dim,
                        dropout=dropout,
                        use_attention=2 ** i in attn_strides,
                    )
                )

                in_channel = channel_mult

            if i != 0:
                up_layers.append(Upsample(in_channel))

        self.up = nn.ModuleList(up_layers)

        # self.out = nn.Sequential(
        #     nn.GroupNorm(32, in_channel),
        #     Swish(),
        #     conv2d(in_channel, 3 * (fold ** 2), 3, padding=1, scale=1e-10),
        # )

        self.out = nn.Sequential(
            nn.GroupNorm(32, in_channel),
            Swish(),
            # conv2d(in_channel, 16, 3, padding=1, scale=1e-10),
            conv2d(in_channel, 3, 3, padding=1, scale=1e-10),
        )

        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_dim)
        #     self.label_emb = nn.Sequential(
        #     # nn.Linear(num_classes, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, time_dim),
        # )

    def forward(self, input, time, cond=None):
        # import pdb; pdb.set_trace()
        time_embed = self.time(time) + self.label_emb(cond)

        feats = []

        out = spatial_fold(input, self.fold)
        for layer in self.down:
            if isinstance(layer, ResBlockWithAttention):
                out = layer(out, time_embed)

            else:
                out = layer(out)

            feats.append(out)

        for layer in self.mid:
            out = layer(out, time_embed)

        for layer in self.up:
            if isinstance(layer, ResBlockWithAttention):
                out = layer(torch.cat((out, feats.pop()), 1), time_embed)

            else:
                out = layer(out)

        out = self.out(out)
        # out = spatial_unfold(out, self.fold)

        return out


class AE_CNN(nn.Module):
    def __init__(
            self,
            in_dim,
            time_step=1000,
    ):
        super().__init__()

        # self.enc1 = nn.Sequential(nn.Conv1d(1, 10, 3, stride=1),nn.LeakyReLU(),nn.Conv1d(1, 10, 3, stride=1),)
        self.enc1 = nn.Sequential(
            nn.InstanceNorm1d(in_dim),
            nn.Conv1d(1, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(in_dim),
            nn.Conv1d(64, 64, 3, stride=1, padding=1),
        )
        self.enc2 = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(in_dim),
            nn.Conv1d(64, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(in_dim),
            nn.Conv1d(64, 128, 3, stride=1, padding=1),
        )
        self.enc3 = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(in_dim),
            nn.Conv1d(128, 128, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(in_dim),
            nn.Conv1d(128, 256, 3, stride=1, padding=1),
        )
        self.enc4 = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(in_dim),
            nn.Conv1d(256, 256, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(in_dim),
            nn.Conv1d(256, 512, 3, stride=1, padding=1),
        )

        self.dec1 = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(in_dim),
            nn.Conv1d(512, 256, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(in_dim),
            nn.Conv1d(256, 256, 3, stride=1, padding=1),
        )
        self.dec2 = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(in_dim),
            nn.Conv1d(256, 128, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(in_dim),
            nn.Conv1d(128, 128, 3, stride=1, padding=1),
        )
        self.dec3 = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(in_dim),
            nn.Conv1d(128, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(in_dim),
            nn.Conv1d(64, 64, 3, stride=1, padding=1),
        )
        self.dec4 = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(in_dim),
            nn.Conv1d(64, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(in_dim),
            nn.Conv1d(64, 1, 3, stride=1, padding=1),
            nn.Tanh(),
        )
        self.time_encode = nn.Embedding(time_step, in_dim)

    def forward(self, input, time, cond=None):
        # import pdb; pdb.set_trace()
        input_shape = input.shape
        time_info = self.time_encode(time)
        time_info = time_info.view(input.size(0), 1, -1)
        if len(input.size()) == 2:
            input = input.view(input.size(0), 1, -1)

        # import pdb; pdb.set_trace()
        emb_enc1 = self.enc1(input + time_info)
        emb_enc2 = self.enc2(emb_enc1 + time_info)
        emb_enc3 = self.enc3(emb_enc2 + time_info)
        emb_enc4 = self.enc4(emb_enc3 + time_info)

        emb_dec1 = self.dec1(emb_enc4 + time_info) + emb_enc3
        emb_dec2 = self.dec2(emb_dec1 + time_info) + emb_enc2
        emb_dec3 = self.dec3(emb_dec2 + time_info) + emb_enc1
        emb_dec4 = self.dec4(emb_dec3 + time_info)

        return emb_dec4.reshape(input_shape)


##
## LDM的backbone(同时也是convAE work for 30k量级size模型 但是有些输入特征维度不同 所以下面还有一个AE_CNN_bottleneck_ori,是convAE的backbone)
class AE_CNN_bottleneck(nn.Module):
    def __init__(
            self,
            in_dim,
            in_channel,
            time_step=1000,
            dec=None
    ):
        super().__init__()

        self.channel_list = [64, 128, 256, 512]  # todo: self.channel_list*2

        # self.enc1 = nn.Sequential(nn.Conv1d(1, 10, 3, stride=1),nn.LeakyReLU(),nn.Conv1d(1, 10, 3, stride=1),)
        self.in_dim = in_dim
        self.fold_rate = 1
        self.kernal_size = 3
        self.dec = dec
        self.real_input_dim = in_dim
        # (
        #     int((in_dim+1000) / self.fold_rate**4 + 1) * self.fold_rate**4
        # )

        self.enc1 = nn.Sequential(
            nn.InstanceNorm1d(self.real_input_dim),
            nn.Conv1d(in_channel, self.channel_list[0], self.kernal_size, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim),
            nn.Conv1d(self.channel_list[0], self.channel_list[0], self.kernal_size, stride=self.fold_rate, padding=1),
            # nn.MaxPool1d(2),
        )
        self.enc2 = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate),
            nn.Conv1d(self.channel_list[0], self.channel_list[1], self.kernal_size, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate),
            nn.Conv1d(self.channel_list[1], self.channel_list[1], self.kernal_size, stride=self.fold_rate, padding=1),
            # nn.MaxPool1d(2),
        )
        self.enc3 = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate ** 2),
            nn.Conv1d(self.channel_list[1], self.channel_list[2], self.kernal_size, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate ** 2),
            nn.Conv1d(self.channel_list[2], self.channel_list[2], self.kernal_size, stride=self.fold_rate, padding=1),
            # nn.MaxPool1d(2),
        )
        self.enc4 = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate ** 3),
            nn.Conv1d(self.channel_list[2], self.channel_list[3], self.kernal_size, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate ** 3),
            nn.Conv1d(self.channel_list[3], self.channel_list[3], self.kernal_size, stride=self.fold_rate, padding=1),
            # nn.MaxPool1d(2),
        )

        self.dec1 = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate ** 4),
            # nn.ConvTranspose1d(
            nn.Conv1d(
                self.channel_list[3], self.channel_list[2], self.kernal_size, stride=self.fold_rate, padding=1
            ),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate ** 4),
            nn.Conv1d(self.channel_list[2], self.channel_list[2], self.kernal_size, stride=1, padding=1),
        )
        self.dec2 = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate ** 3),
            # nn.ConvTranspose1d(
            nn.Conv1d(
                self.channel_list[2], self.channel_list[1], self.kernal_size, stride=self.fold_rate, padding=1
            ),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate ** 3),
            nn.Conv1d(self.channel_list[1], self.channel_list[1], self.kernal_size, stride=1, padding=1),
        )
        self.dec3 = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate ** 2),
            # nn.ConvTranspose1d(
            nn.Conv1d(
                self.channel_list[1], self.channel_list[0], self.kernal_size, stride=self.fold_rate, padding=1
            ),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate ** 2),
            nn.Conv1d(self.channel_list[0], self.channel_list[0], self.kernal_size, stride=1, padding=1),
        )
        self.dec4 = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate),
            # nn.ConvTranspose1d(
            nn.Conv1d(
                self.channel_list[0], self.channel_list[0], self.kernal_size, stride=self.fold_rate, padding=1
            ),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate),
            nn.Conv1d(self.channel_list[0], in_channel, self.kernal_size, stride=1, padding=1),
        )

        self.time_encode = nn.Embedding(time_step, self.real_input_dim)

    def forward(self, input, time, cond=None):
        assert input.shape[1] * input.shape[2] == self.in_dim

        input_shape = input.shape
        input = input.reshape(input.shape[0], 1, -1)
        # import pdb;pdb.set_trace()
        time_info = self.time_encode(time)[0, None, None]
        time_info = time_info.repeat((input.shape[0], 1, 1))

        if len(input.size()) == 2:
            input = input.view(input.size(0), 1, -1)

        input = input
        # input = torch.cat(
        #     [
        #         input,
        #         # time_info.repeat((1,input.shape[1],1)),
        #         torch.zeros(input.shape[0], input.shape[1], (self.real_input_dim - self.in_dim)).to(
        #             input.device
        #         ),
        #     ],
        #     dim=2,
        # )
        # import pdb; pdb.set_trace()
        # time_info = torch.cat([time_info, torch.zeros(time_info.shape[0],1,6).to(time_info.device) ], dim=2)
        emb_enc1 = self.enc1(input + time_info)
        emb_enc2 = self.enc2(emb_enc1 + time_info)
        emb_enc3 = self.enc3(emb_enc2 + time_info)
        emb_enc4 = self.enc4(emb_enc3 + time_info)
        # import pdb; pdb.set_trace()

        emb_dec1 = self.dec1(emb_enc4 + time_info) + emb_enc3
        emb_dec2 = self.dec2(emb_dec1 + time_info) + emb_enc2
        emb_dec3 = self.dec3(emb_dec2 + time_info) + emb_enc1
        emb_dec4 = self.dec4(emb_dec3 + time_info)
        # import pdb; pdb.set_trace()

        emb_dec4 = emb_dec4.reshape(input_shape)
        # if self.dec is not None:
        #     emb_dec4 = self.dec.Dec(emb_dec4)

        return emb_dec4


class AE_CNN_bottleneck2(nn.Module):
    def __init__(
            self,
            in_dim,
            in_channel,
            time_step=1000,
            dec=None
    ):
        super().__init__()

        self.channel_list = [64, 128, 256, 512]  # todo: self.channel_list*2
        self.channel_list = [i * 2 for i in self.channel_list]
        print('diffusion ae channel list', self.channel_list)
        # self.enc1 = nn.Sequential(nn.Conv1d(1, 10, 3, stride=1),nn.LeakyReLU(),nn.Conv1d(1, 10, 3, stride=1),)
        self.in_dim = in_dim
        self.fold_rate = 1
        self.kernal_size = 3
        self.dec = dec
        self.real_input_dim = in_dim
        # (
        #     int((in_dim+1000) / self.fold_rate**4 + 1) * self.fold_rate**4
        # )

        self.enc1 = nn.Sequential(
            nn.InstanceNorm1d(self.real_input_dim),
            nn.Conv1d(in_channel, self.channel_list[0], self.kernal_size, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim),
            nn.Conv1d(self.channel_list[0], self.channel_list[0], self.kernal_size, stride=self.fold_rate, padding=1),
            # nn.MaxPool1d(2),
        )
        self.enc2 = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate),
            nn.Conv1d(self.channel_list[0], self.channel_list[1], self.kernal_size, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate),
            nn.Conv1d(self.channel_list[1], self.channel_list[1], self.kernal_size, stride=self.fold_rate, padding=1),
            # nn.MaxPool1d(2),
        )
        self.enc3 = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate ** 2),
            nn.Conv1d(self.channel_list[1], self.channel_list[2], self.kernal_size, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate ** 2),
            nn.Conv1d(self.channel_list[2], self.channel_list[2], self.kernal_size, stride=self.fold_rate, padding=1),
            # nn.MaxPool1d(2),
        )
        self.enc4 = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate ** 3),
            nn.Conv1d(self.channel_list[2], self.channel_list[3], self.kernal_size, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate ** 3),
            nn.Conv1d(self.channel_list[3], self.channel_list[3], self.kernal_size, stride=self.fold_rate, padding=1),
            # nn.MaxPool1d(2),
        )

        self.dec1 = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate ** 4),
            # nn.ConvTranspose1d(
            nn.Conv1d(
                self.channel_list[3], self.channel_list[2], self.kernal_size, stride=self.fold_rate, padding=1
            ),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate ** 4),
            nn.Conv1d(self.channel_list[2], self.channel_list[2], self.kernal_size, stride=1, padding=1),
        )
        self.dec2 = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate ** 3),
            # nn.ConvTranspose1d(
            nn.Conv1d(
                self.channel_list[2], self.channel_list[1], self.kernal_size, stride=self.fold_rate, padding=1
            ),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate ** 3),
            nn.Conv1d(self.channel_list[1], self.channel_list[1], self.kernal_size, stride=1, padding=1),
        )
        self.dec3 = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate ** 2),
            # nn.ConvTranspose1d(
            nn.Conv1d(
                self.channel_list[1], self.channel_list[0], self.kernal_size, stride=self.fold_rate, padding=1
            ),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate ** 2),
            nn.Conv1d(self.channel_list[0], self.channel_list[0], self.kernal_size, stride=1, padding=1),
        )
        self.dec4 = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate),
            # nn.ConvTranspose1d(
            nn.Conv1d(
                self.channel_list[0], self.channel_list[0], self.kernal_size, stride=self.fold_rate, padding=1
            ),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate),
            nn.Conv1d(self.channel_list[0], in_channel, self.kernal_size, stride=1, padding=1),
        )

        self.time_encode = nn.Embedding(time_step, self.real_input_dim)

    def forward(self, input, time, cond=None):
        # if self.dec is not None:
        #     input = self.dec.Enc(input).detach()
        assert input.shape[1] * input.shape[2] == self.in_dim

        input_shape = input.shape
        input = input.reshape(input.shape[0], 1, -1)
        # import pdb;pdb.set_trace()
        time_info = self.time_encode(time)[0, None, None]
        time_info = time_info.repeat((input.shape[0], 1, 1))

        if len(input.size()) == 2:
            input = input.view(input.size(0), 1, -1)

        input = input
        # input = torch.cat(
        #     [
        #         input,
        #         # time_info.repeat((1,input.shape[1],1)),
        #         torch.zeros(input.shape[0], input.shape[1], (self.real_input_dim - self.in_dim)).to(
        #             input.device
        #         ),
        #     ],
        #     dim=2,
        # )
        # import pdb; pdb.set_trace()
        # time_info = torch.cat([time_info, torch.zeros(time_info.shape[0],1,6).to(time_info.device) ], dim=2)
        emb_enc1 = self.enc1(input + time_info)
        emb_enc2 = self.enc2(emb_enc1 + time_info)
        emb_enc3 = self.enc3(emb_enc2 + time_info)
        emb_enc4 = self.enc4(emb_enc3 + time_info)
        # import pdb; pdb.set_trace()

        emb_dec1 = self.dec1(emb_enc4 + time_info) + emb_enc3
        emb_dec2 = self.dec2(emb_dec1 + time_info) + emb_enc2
        emb_dec3 = self.dec3(emb_dec2 + time_info) + emb_enc1
        emb_dec4 = self.dec4(emb_dec3 + time_info)
        # import pdb; pdb.set_trace()

        emb_dec4 = emb_dec4.reshape(input_shape)
        # if self.dec is not None:
        #     emb_dec4 = self.dec.Dec(emb_dec4)

        return emb_dec4


# for cnnAE
class AE_CNN_bottleneck_original(nn.Module):
    def __init__(
            self,
            in_dim,
            time_step=1000,
    ):
        super().__init__()

        # self.enc1 = nn.Sequential(nn.Conv1d(1, 10, 3, stride=1),nn.LeakyReLU(),nn.Conv1d(1, 10, 3, stride=1),)
        self.in_dim = in_dim
        self.real_input_dim = int(in_dim / 3 ** 4 + 1) * 3 ** 4

        self.enc1 = nn.Sequential(
            nn.InstanceNorm1d(self.real_input_dim),
            nn.Conv1d(1, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim),
            nn.Conv1d(64, 64, 3, stride=3, padding=0),
            # nn.MaxPool1d(2),
        )
        self.enc2 = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // 2),
            nn.Conv1d(64, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // 2),
            nn.Conv1d(64, 128, 3, stride=3, padding=0),
            # nn.MaxPool1d(2),
        )
        self.enc3 = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // 4),
            nn.Conv1d(128, 128, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // 4),
            nn.Conv1d(128, 256, 3, stride=3, padding=0),
            # nn.MaxPool1d(2),
        )
        self.enc4 = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // 8),
            nn.Conv1d(256, 256, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // 8),
            nn.Conv1d(256, 512, 3, stride=3, padding=0),
            # nn.MaxPool1d(2),
        )

        self.dec1 = nn.Sequential(nn.LeakyReLU(), nn.InstanceNorm1d(self.real_input_dim // 16),
                                  nn.Conv1d(512, 256, 3, stride=1, padding=1), nn.LeakyReLU(),
                                  nn.InstanceNorm1d(self.real_input_dim // 16),
                                  nn.ConvTranspose1d(256, 256, 3, stride=3, padding=0), )
        self.dec2 = nn.Sequential(nn.LeakyReLU(), nn.InstanceNorm1d(self.real_input_dim // 8),
                                  nn.Conv1d(256, 128, 3, stride=1, padding=1), nn.LeakyReLU(),
                                  nn.InstanceNorm1d(self.real_input_dim // 8),
                                  nn.ConvTranspose1d(128, 128, 3, stride=3, padding=0), )
        self.dec3 = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // 4),
            nn.Conv1d(128, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // 4),
            nn.ConvTranspose1d(64, 64, 3, stride=3, padding=0),
        )
        self.dec4 = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // 2),
            nn.Conv1d(64, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // 2),
            # nn.Conv1d(64, 1, 3, stride=1, padding=1),
            nn.ConvTranspose1d(64, 1, 3, stride=3, padding=0),
        )
        self.time_encode = nn.Embedding(time_step, self.real_input_dim)

    def forward(self, input, time, cond=None):
        input_shape = input.shape
        time_info = self.time_encode(time)
        time_info = time_info.view(input.size(0), 1, -1)
        if len(input.size()) == 2:
            input = input.view(input.size(0), 1, -1)

        input = torch.cat([input, torch.zeros(input.shape[0], 1, (self.real_input_dim - self.in_dim)).to(input.device)],
                          dim=2)
        # time_info = torch.cat([time_info, torch.zeros(time_info.shape[0],1,6).to(time_info.device) ], dim=2)

        # import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()
        emb_enc1 = self.enc1(input + time_info)
        emb_enc2 = self.enc2(emb_enc1)
        emb_enc3 = self.enc3(emb_enc2)
        emb_enc4 = self.enc4(emb_enc3)

        # import pdb; pdb.set_trace()
        emb_dec1 = self.dec1(emb_enc4) + emb_enc3
        emb_dec2 = self.dec2(emb_dec1) + emb_enc2
        emb_dec3 = self.dec3(emb_dec2) + emb_enc1
        emb_dec4 = self.dec4(emb_dec3)

        emb_dec4 = emb_dec4[:, :, :self.in_dim]

        return emb_dec4.reshape(input_shape)


class AE_CNN_bottleneck_deep(nn.Module):
    def __init__(
            self,
            in_dim,
            time_step=1000,
    ):
        super().__init__()

        # self.enc1 = nn.Sequential(nn.Conv1d(1, 10, 3, stride=1),nn.LeakyReLU(),nn.Conv1d(1, 10, 3, stride=1),)
        self.in_dim = in_dim
        self.real_input_dim = int(in_dim / 3 ** 4 + 1) * 3 ** 4

        self.enc1 = nn.Sequential(
            nn.InstanceNorm1d(self.real_input_dim),
            nn.Conv1d(1, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim),
            nn.Conv1d(64, 128, 3, stride=3, padding=0),
            # nn.MaxPool1d(2),
        )
        self.enc2 = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // 2),
            nn.Conv1d(128, 128, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // 2),
            nn.Conv1d(128, 256, 3, stride=3, padding=0),
            # nn.MaxPool1d(2),
        )
        self.enc3 = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // 4),
            nn.Conv1d(256, 256, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // 4),
            nn.Conv1d(256, 512, 3, stride=3, padding=0),
            # nn.MaxPool1d(2),
        )
        self.enc4 = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // 8),
            nn.Conv1d(512, 512, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // 8),
            nn.Conv1d(512, 1024, 3, stride=3, padding=0),
            # nn.MaxPool1d(2),
        )

        self.dec1 = nn.Sequential(nn.LeakyReLU(), nn.InstanceNorm1d(self.real_input_dim // 16),
                                  nn.Conv1d(1024, 512, 3, stride=1, padding=1), nn.LeakyReLU(),
                                  nn.InstanceNorm1d(self.real_input_dim // 16),
                                  nn.ConvTranspose1d(512, 512, 3, stride=3, padding=0), )
        self.dec2 = nn.Sequential(nn.LeakyReLU(), nn.InstanceNorm1d(self.real_input_dim // 8),
                                  nn.Conv1d(512, 256, 3, stride=1, padding=1), nn.LeakyReLU(),
                                  nn.InstanceNorm1d(self.real_input_dim // 8),
                                  nn.ConvTranspose1d(256, 256, 3, stride=3, padding=0), )
        self.dec3 = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // 4),
            nn.Conv1d(256, 128, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // 4),
            nn.ConvTranspose1d(128, 128, 3, stride=3, padding=0),
        )
        self.dec4 = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // 2),
            nn.Conv1d(128, 128, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // 2),
            # nn.Conv1d(64, 1, 3, stride=1, padding=1),
            nn.ConvTranspose1d(128, 1, 3, stride=3, padding=0),
        )
        self.time_encode = nn.Embedding(time_step, self.real_input_dim)

    def forward(self, input, time, cond=None):
        input_shape = input.shape
        time_info = self.time_encode(time)
        time_info = time_info.view(input.size(0), 1, -1)
        if len(input.size()) == 2:
            input = input.view(input.size(0), 1, -1)

        input = torch.cat([input, torch.zeros(input.shape[0], 1, (self.real_input_dim - self.in_dim)).to(input.device)],
                          dim=2)
        # time_info = torch.cat([time_info, torch.zeros(time_info.shape[0],1,6).to(time_info.device) ], dim=2)

        # import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()
        emb_enc1 = self.enc1(input + time_info)
        emb_enc2 = self.enc2(emb_enc1)
        emb_enc3 = self.enc3(emb_enc2)
        emb_enc4 = self.enc4(emb_enc3)

        # import pdb; pdb.set_trace()
        emb_dec1 = self.dec1(emb_enc4) + emb_enc3
        emb_dec2 = self.dec2(emb_dec1) + emb_enc2
        emb_dec3 = self.dec3(emb_dec2) + emb_enc1
        emb_dec4 = self.dec4(emb_dec3)

        emb_dec4 = emb_dec4[:, :, :self.in_dim]

        return emb_dec4.reshape(input_shape)


class AE_CNN_bottleneck_ori(nn.Module):
    def __init__(
            self,
            in_dim,
            in_channel,
            time_step=1000,
            dec=None
    ):
        super().__init__()

        self.channel_list = [64, 128, 256, 512]

        # self.enc1 = nn.Sequential(nn.Conv1d(1, 10, 3, stride=1),nn.LeakyReLU(),nn.Conv1d(1, 10, 3, stride=1),)
        self.in_dim = in_dim
        self.fold_rate = 1
        self.kernal_size = 3
        self.dec = dec
        self.real_input_dim = in_dim
        # (
        #     int((in_dim+1000) / self.fold_rate**4 + 1) * self.fold_rate**4
        # )

        self.enc1 = nn.Sequential(
            nn.InstanceNorm1d(self.real_input_dim),
            nn.Conv1d(in_channel, self.channel_list[0], self.kernal_size, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim),
            nn.Conv1d(self.channel_list[0], self.channel_list[0], self.kernal_size, stride=self.fold_rate, padding=1),
            # nn.MaxPool1d(2),
        )
        self.enc2 = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate),
            nn.Conv1d(self.channel_list[0], self.channel_list[1], self.kernal_size, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate),
            nn.Conv1d(self.channel_list[1], self.channel_list[1], self.kernal_size, stride=self.fold_rate, padding=1),
            # nn.MaxPool1d(2),
        )
        self.enc3 = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate ** 2),
            nn.Conv1d(self.channel_list[1], self.channel_list[2], self.kernal_size, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate ** 2),
            nn.Conv1d(self.channel_list[2], self.channel_list[2], self.kernal_size, stride=self.fold_rate, padding=1),
            # nn.MaxPool1d(2),
        )
        self.enc4 = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate ** 3),
            nn.Conv1d(self.channel_list[2], self.channel_list[3], self.kernal_size, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate ** 3),
            nn.Conv1d(self.channel_list[3], self.channel_list[3], self.kernal_size, stride=self.fold_rate, padding=1),
            # nn.MaxPool1d(2),
        )

        self.dec1 = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate ** 4),
            # nn.ConvTranspose1d(
            nn.Conv1d(
                self.channel_list[3], self.channel_list[2], self.kernal_size, stride=self.fold_rate, padding=1
            ),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate ** 4),
            nn.Conv1d(self.channel_list[2], self.channel_list[2], self.kernal_size, stride=1, padding=1),
        )
        self.dec2 = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate ** 3),
            # nn.ConvTranspose1d(
            nn.Conv1d(
                self.channel_list[2], self.channel_list[1], self.kernal_size, stride=self.fold_rate, padding=1
            ),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate ** 3),
            nn.Conv1d(self.channel_list[1], self.channel_list[1], self.kernal_size, stride=1, padding=1),
        )
        self.dec3 = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate ** 2),
            # nn.ConvTranspose1d(
            nn.Conv1d(
                self.channel_list[1], self.channel_list[0], self.kernal_size, stride=self.fold_rate, padding=1
            ),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate ** 2),
            nn.Conv1d(self.channel_list[0], self.channel_list[0], self.kernal_size, stride=1, padding=1),
        )
        self.dec4 = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate),
            # nn.ConvTranspose1d(
            nn.Conv1d(
                self.channel_list[0], self.channel_list[0], self.kernal_size, stride=self.fold_rate, padding=1
            ),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(self.real_input_dim // self.fold_rate),
            nn.Conv1d(self.channel_list[0], in_channel, self.kernal_size, stride=1, padding=1),
        )

        self.time_encode = nn.Embedding(time_step, self.real_input_dim)

    def forward(self, input, time, cond=None):
        # if self.dec is not None:
        #     input = self.dec.Enc(input).detach()
        # assert input.shape[1]*input.shape[2] == self.in_dim

        input_shape = input.shape
        input = input.reshape(input.shape[0], 1, -1)
        time_info = self.time_encode(time)[1, None, None]
        time_info = time_info.repeat((input.shape[0], 1, 1))

        if len(input.size()) == 2:
            input = input.view(input.size(0), 1, -1)

        input = input
        # input = torch.cat(
        #     [
        #         input,
        #         # time_info.repeat((1,input.shape[1],1)),
        #         torch.zeros(input.shape[0], input.shape[1], (self.real_input_dim - self.in_dim)).to(
        #             input.device
        #         ),
        #     ],
        #     dim=2,
        # )
        # import pdb; pdb.set_trace()
        # time_info = torch.cat([time_info, torch.zeros(time_info.shape[0],1,6).to(time_info.device) ], dim=2)
        emb_enc1 = self.enc1(input + time_info)
        emb_enc2 = self.enc2(emb_enc1 + time_info)
        emb_enc3 = self.enc3(emb_enc2 + time_info)
        emb_enc4 = self.enc4(emb_enc3 + time_info)
        # import pdb; pdb.set_trace()

        emb_dec1 = self.dec1(emb_enc4 + time_info) + emb_enc3
        emb_dec2 = self.dec2(emb_dec1 + time_info) + emb_enc2
        emb_dec3 = self.dec3(emb_dec2 + time_info) + emb_enc1
        emb_dec4 = self.dec4(emb_dec3 + time_info)
        # import pdb; pdb.set_trace()

        emb_dec4 = emb_dec4.reshape(input_shape)
        # if self.dec is not None:
        #     emb_dec4 = self.dec.Dec(emb_dec4)

        return emb_dec4


## backbone fc 不降维 对于小参数量work:9914
class AE(nn.Module):
    def __init__(
            self,
            in_dim,
            time_step=1000,
    ):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(in_dim),
            nn.Linear(in_dim, in_dim),
        )
        self.enc2 = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(in_dim),
            nn.Linear(in_dim, in_dim),
        )
        self.enc3 = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(in_dim),
            nn.Linear(in_dim, in_dim),
        )
        self.enc4 = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(in_dim),
            nn.Linear(in_dim, in_dim),
        )

        self.dec1 = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(in_dim),
            nn.Linear(in_dim, in_dim),
        )
        self.dec2 = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(in_dim),
            nn.Linear(in_dim, in_dim),
        )
        self.dec3 = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(in_dim),
            nn.Linear(in_dim, in_dim),
        )
        self.dec4 = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(in_dim),
            nn.Linear(in_dim, in_dim),
        )
        self.time_encode = nn.Embedding(time_step, in_dim)

    def forward(self, input, time, cond=None):
        # import pdb; pdb.set_trace()
        input_shape = input.shape
        # import pdb;pdb.set_trace()
        if len(input.size()) > 2:
            input = input.view(input.size(0), -1)
        time_info = self.time_encode(time)

        emb_enc1 = self.enc1(input + time_info)
        emb_enc2 = self.enc2(emb_enc1 + time_info) + emb_enc1
        emb_enc3 = self.enc3(emb_enc2 + time_info) + emb_enc1 + emb_enc2
        emb_enc4 = self.enc4(emb_enc3 + time_info) + emb_enc1 + emb_enc2 + emb_enc3

        emb_dec1 = self.dec1(emb_enc4 + time_info)
        emb_dec2 = self.dec2(emb_dec1 + time_info) + emb_enc3 + emb_dec1
        emb_dec3 = self.dec3(emb_dec2 + time_info) + emb_enc2 + emb_dec1 + emb_dec2
        emb_dec4 = (
                self.dec4(emb_dec3 + time_info) + emb_enc1 + emb_dec1 + emb_dec2 + emb_dec3
        )

        return emb_dec4.reshape(input_shape)


class TF(nn.Module):
    def __init__(
            self,
            in_dim,
            channel=1,
            time_step=1000,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.len_token = 30

        self.enc = TransformerEncoder(
            d_model=self.len_token,
            nhead=5,
            dim_feedforward=2048,
            num_layers=3,
            dropout=0.1,
        )
        self.dec = TransformerDecoder(
            d_model=self.len_token,
            nhead=5,
            dim_feedforward=2048,
            num_layers=3,
            dropout=0.1,
        )
        self.time_encode = nn.Sequential(
            # TimeEmbedding(channel),
            nn.Embedding(time_step, self.len_token),
            # linear(channel, in_dim),
            # Swish(),
            # linear(in_dim, in_dim),
        )

    def forward(self, input, time, cond=None):
        # import pdb; pdb.set_trace()
        # time_embed = self.time(time) # + self.label_emb(cond)
        input_shape = input.shape
        if len(input.size()) > 2:
            input = input.view(input.size(0), -1)
        # print('input.shape', input.shape)
        input_view_shape = input.shape

        # if input_view_shape[1] < self.in_dim:
        input_pad = torch.cat(
            [
                input,
                torch.zeros(input_view_shape[0], self.in_dim - input_view_shape[1]).to(
                    input.device
                ),
            ],
            dim=1,
        )
        input_seq = input_pad.reshape(input_view_shape[0], -1, 30)

        # import pdb; pdb.set_trace()
        time_emb = self.time_encode(time)
        time_emb_rs = time_emb.reshape(input_view_shape[0], 1, 30)
        emb_enc1 = self.enc(input_seq + time_emb_rs)
        # print('time_emb.shape', time_emb.shape)
        # print('emb_enc1.shape', emb_enc1.shape)
        out_dec1 = self.dec(emb_enc1, emb_enc1)

        # if input_view_shape[1] < self.in_dim:
        out = out_dec1.reshape(input_view_shape[0], -1)
        out = out[:, : input_view_shape[1]]
        out = out.reshape(input_shape)

        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    def __init__(self, nhead, dim_feedforward, num_layers, d_model, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.pos_encoder = PositionalEncoding(d_model)

    def forward(self, src):
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        return output


class TransformerDecoder(nn.Module):
    def __init__(self, nhead, dim_feedforward, num_layers, d_model, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        decoder_layer = nn.TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.pos_decoder = PositionalEncoding(d_model)

    def forward(self, tgt, memory):
        tgt = self.pos_decoder(tgt)
        output = self.transformer_decoder(tgt, memory)
        return output
