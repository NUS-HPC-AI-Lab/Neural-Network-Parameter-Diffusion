from torch import nn
import torch
#todo: kaiwang, core.diffusion.u_net, starting from AE_CNN; core.module.encoder, base on core.encoder.autoencoder
class ODUnetEncoder(nn.Module):
    def __init__(self, in_dim_list, enc_channel_list, fold_rate, kernal_size):
        super(ODUnetEncoder, self).__init__()
        self.in_dim_list = in_dim_list
        self.enc_channel_list = enc_channel_list
        self.fold_rate = fold_rate
        self.kernal_size = kernal_size
        encoder = nn.ModuleList()
        layer_num = len(in_dim_list)
        for i in range(layer_num):
            layer = self.build_layer(in_dim_list[i], enc_channel_list[i], enc_channel_list[i+1], kernal_size, fold_rate)
            encoder.append(layer)
        self.encoder = encoder

    def build_layer(self, in_dim, in_channel, out_channel, kernel_size, fold_rate):
        layer = nn.Sequential(
            nn.InstanceNorm1d(in_dim),
            nn.Conv1d(in_channel, out_channel, kernel_size, padding=1, stride=1),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(in_dim),
            nn.Conv1d(out_channel, out_channel, kernel_size, padding=1, stride=fold_rate),
            nn.LeakyReLU()
        )
        return layer

    def forward(self, x, time_info, **kwargs):
        encoder_output = []
        for layer in self.encoder:
            x = layer(x + time_info)
            encoder_output.append(x)
        return encoder_output


class ODUnetDecoder(nn.Module):
    def __init__(self, in_dim_list, dec_channel_list, fold_rate, kernal_size):
        super(ODUnetDecoder, self).__init__()
        self.in_dim_list =  in_dim_list
        self.dec_channel_list = dec_channel_list
        self.fold_rate = fold_rate
        self.kernal_size = kernal_size

        decoder = nn.ModuleList()
        layer_num = len(in_dim_list)
        for i in range(layer_num):
            layer = self.build_layer(in_dim_list[i], dec_channel_list[i], dec_channel_list[i+1], kernal_size, fold_rate, last=(i==layer_num-1))
            decoder.append(layer)
        self.decoder = decoder

    def build_layer(self, in_dim, in_channel, out_channel, kernel_size, fold_rate, last=False):
        layer = nn.Sequential(
            nn.InstanceNorm1d(in_dim),
            nn.Conv1d(in_channel, out_channel, kernel_size, padding=fold_rate, stride=1),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(in_dim),
            nn.Conv1d(out_channel, out_channel, kernel_size, padding=1, stride=1),
            nn.LeakyReLU() if not last else nn.Identity(),
        )
        return layer

    def forward(self, x, time_info, **kwargs):
        y = x[-1]
        for i, layer in enumerate(self.decoder):
            if i == len(self.decoder) - 1:
                y = layer(y + time_info)
                break
            y = layer(y + time_info) + x[2-i]
        return x





class ODUnet(nn.Module):
    def __init__(self, in_dim, in_channel=1, time_step=1000, enc_channel_list=None, dec_channel_list=None, fold_rate=1, kernal_size=3):
        super(ODUnet, self).__init__()

        if enc_channel_list is None:
            enc_channel_list = [64, 128, 256, 512]
        if dec_channel_list is None:
            dec_channel_list = [512, 256, 128, 64]

        enc_dim_list = [in_dim // (fold_rate ** i) for i in range(len(enc_channel_list))]
        dec_dim_list = [in_dim // (fold_rate ** (4 - i)) for i in range(len(dec_channel_list))]

        self.in_dim = in_dim

        enc_channel_list = [in_channel] + enc_channel_list
        dec_channel_list = dec_channel_list + [in_channel]


        self.encoder = ODUnetEncoder(enc_dim_list, enc_channel_list, fold_rate, kernal_size)
        self.deoder = ODUnetDecoder(dec_dim_list, dec_channel_list, fold_rate, kernal_size)
        self.time_encode = nn.Embedding(time_step, self.in_dim)



    def adjust_input(self, input, time):
        assert input.shape[1] * input.shape[2] == self.in_dim

        self.input_shape = input.shape
        input = input.reshape(input.shape[0], 1, -1)
        # import pdb;pdb.set_trace()
        time_info = self.time_encode(time)[0, None, None]
        time_info = time_info.repeat((input.shape[0], 1, 1))

        if len(input.size()) == 2:
            input = input.view(input.size(0), 1, -1)

        input = input
        return input,time_info

    def forward(self, input, time, **kwargs):
        x, time_info = self.encode(input, time)
        x = self.decode(x, time_info)


        x = torch.clamp(x, -1, 1)

        return x

    def encode(self, x, time, **kwargs):
        input, time_info = self.adjust_input(x, time)
        enc_output = self.encoder(input, time_info, **kwargs)
        return enc_output, time_info

    def decode(self, x, time_info, **kwargs):
        dec_output =  self.decoder(x, time_info, **kwargs)
        return dec_output.reshape(self.input_shape)

class AE_CNN_bottleneck(ODUnet):
    def __init__(self, in_dim):
        enc_channel_list = [64, 128, 256, 512]
        fold_rate = 1
        kernal_size = 3
        in_channel = 1
        super(AE_CNN_bottleneck, self).__init__(in_dim, in_channel, enc_channel_list=enc_channel_list, fold_rate=fold_rate, kernal_size=kernal_size,)
