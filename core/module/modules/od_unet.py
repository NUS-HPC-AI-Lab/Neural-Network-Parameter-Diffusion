from torch import nn

#todo: kaiwang, core.diffusion.u_net, starting from AE_CNN; core.module.encoder, base on core.encoder.autoencoder
class ODUnetEncoder(nn.Module):
    def __init__(self, in_dim, enc_channel_list, fold_rate, kernal_size, time_step):
        super(ODUnetEncoder, self).__init__()
        self.in_dim = in_dim
        self.enc_channel_list = enc_channel_list
        self.fold_rate = fold_rate
        self.kernal_size = kernal_size
        self.time_step = time_step
        real_input_dim = (
                int(in_dim / self.fold_rate ** 4 + 1) * self.fold_rate ** 4
        )
        self.real_input_dim = real_input_dim

        enc_channel_list = [enc_channel_list[0]] + enc_channel_list
        self.enc_channel_list = enc_channel_list
        encoder = nn.ModuleList()
        layer_num = len(enc_channel_list) - 1

        for i in range(layer_num):
            ratio = i
            if i == 0:
                if_first = True, if_last = False
                layer = self.build_encoder_layer(enc_channel_list[i], enc_channel_list[i + 1], fold_rate, kernal_size,
                                                 real_input_dim, ratio, if_first, if_last)
            elif i == layer_num - 1:
                if_first = False, if_last = True
                layer = self.build_encoder_layer(enc_channel_list[i], enc_channel_list[i + 1], fold_rate, kernal_size,
                                                 real_input_dim, ratio, if_first, if_last)
            else:
                if_first = False, if_last = False
                layer = self.build_encoder_layer(enc_channel_list[i], enc_channel_list[i + 1], fold_rate, kernal_size,
                                                 real_input_dim, ratio, if_first, if_last)
            encoder.append(layer)
        self.encoder = encoder
        self.time_encode = nn.Embedding(time_step, self.real_input_dim)

    def build_encoder_layer(self, enc_input_channel_list, enc_output_channel_list, fold_rate, kernal_size,
                            real_input_dim, ratio, first=False, last=False):
        # first: if is the first layer of encoder

        layer = nn.Sequential(
            nn.InstanceNorm1d(real_input_dim) if first else nn.InstanceNorm1d(real_input_dim // fold_rate ** ratio),
            nn.Conv1d(enc_input_channel_list, enc_input_channel_list, kernal_size, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(real_input_dim) if first else nn.InstanceNorm1d(real_input_dim // fold_rate ** ratio),
            nn.Conv1d(enc_input_channel_list, enc_output_channel_list, kernal_size, stride=fold_rate, padding=0),
            nn.Tanh() if last else nn.LeakyReLU(),
        )
        return layer

    def forward(self, x, time, **kwargs):
        result = []
        time_info = self.time_encode(time)[0, None, None]
        time_info = time_info.repeat((x.shape[0], 1, 1))
        for i, module in enumerate(self.encoder):
            x = module(x + time_info)
        return result.append(x), time_info

class ODUnetDecoder(nn.Module):
    def __init__(self, in_dim, dec_channel_list, fold_rate, kernal_size):
        super(ODUnetDecoder, self).__init__()
        self.in_dim = in_dim
        self.dec_channel_list = dec_channel_list
        self.fold_rate = fold_rate
        self.kernal_size = kernal_size
        real_input_dim = (
                int(in_dim / self.fold_rate ** 4 + 1) * self.fold_rate ** 4
        )
        self.real_input_dim = real_input_dim

        dec_channel_list = dec_channel_list + [dec_channel_list[-1]]
        self.dec_channel_list = dec_channel_list
        decoder = nn.ModuleList()
        layer_num = len(dec_channel_list) - 1

        for i in range(layer_num):
            ratio = layer_num - i
            layer = self.build_decoder_layer(dec_channel_list[i], dec_channel_list[i + 1], fold_rate, kernal_size,
                                             real_input_dim, ratio, if_last)
            if i == layer_num - 1:
                if_last = True
                layer = self.build_decoder_layer(dec_channel_list[i], dec_channel_list[i + 1], fold_rate, kernal_size,
                                                 real_input_dim, ratio, if_last)
            decoder.append(layer)
        self.decoder = decoder

    def build_decoder_layer(self, dec_input_channel_list, dec_output_channel_list, fold_rate, kernal_size,
                            real_input_dim, ratio, last=False):

        layer = nn.Sequential(
            nn.LeakyReLU(),
            nn.InstanceNorm1d(real_input_dim // fold_rate ** ratio),
            nn.ConvTranspose1d(
                dec_input_channel_list, dec_input_channel_list, kernal_size, stride=fold_rate, padding=0
            ),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(real_input_dim // fold_rate ** ratio),
            nn.Conv1d(dec_input_channel_list, 1, kernal_size, stride=1, padding=fold_rate) if last else nn.Conv1d(
                dec_input_channel_list, dec_output_channel_list, kernal_size, stride=1, padding=fold_rate - 1),
        )
        return layer

    def forward(self, x, time_info, **kwargs):
        x_dec = x[-1]
        for i, module in enumerate(self.decoder):
            if i+2 > len(x):
                x_dec = module(x_dec + time_info)
            else:
                x_dec = module(x_dec + time_info) + x[-(i+2)]
        return x_dec


class ODUnet(nn.Module):
    def __init__(self, in_dim, enc_channel_list=None, dec_channel_list=None, fold_rate, kernal_size):
        super(ODUnet, self).__init__()

        self.encoder = ODUnetEncoder(in_dim, enc_channel_list, fold_rate, kernal_size)
        self.deoder = ODUnetDecoder(in_dim, dec_channel_list, fold_rate, kernal_size)



    def adjust_input(self, input):
        # time_info = self.time_encode(time)
        # time_info = time_info.view(input.size(0), 1, -1)
        if len(input.size()) == 2:
            input = input.view(input.size(0), 1, -1)
        return input

    def forward(self, input, **kwargs):
        x, time_info = self.encode(input)
        ############latent noise adding###################
        noise_factor = 0.1
        x[-1] = x[-1] + torch.randn_like(x[-1]) * noise_factor
        x[-1] = torch.clamp(x[-1], -1, 1)###或许这个不需要了？
        ##################################################
        x = self.decode(x, time_info)
        x = x[:, :, :input.shape[-1]]

        return x.reshape(input.shape)

    def encode(self, x, **kwargs):
        x = self.adjust_input(x)
        return self.encoder(x, **kwargs)

    def decode(self, x, time_info, **kwargs):
        return self.decoder(x, time_info, **kwargs)

class AE_CNN_bottleneck(ODUnet):
    def __init__(self, ):