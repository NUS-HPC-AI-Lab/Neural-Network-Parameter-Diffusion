from .diffusion import DDIMSampler, DDPMSampler, GaussianDiffusionTrainer
from .denoiser import OneDimCNN
from torch.nn import functional as F
from abc import abstractmethod
from torch import nn
import torch




class PDiff(nn.Module):
    config = {}

    def __init__(self, sequence_length):
        super().__init__()
        self.sequence_length = sequence_length
        self.net = OneDimCNN(
            layer_channels=self.config["layer_channels"],
            model_dim=self.config["model_dim"],
            kernel_size=self.config["kernel_size"],
        )
        self.diffusion_trainer = GaussianDiffusionTrainer(
            model=self.net,
            beta=self.config["beta"],
            T=self.config["T"]
        )
        self.diffusion_sampler = self.config["sample_mode"](
            model=self.net,
            beta=self.config["beta"],
            T=self.config["T"]
        )

    def forward(self, x=None, c=0., **kwargs):
        if kwargs.get("sample"):
            del kwargs["sample"]
            return self.sample(x, c, **kwargs)
        x = x.view(-1, x.size(-1))
        loss = self.diffusion_trainer(x, c, **kwargs)
        return loss

    @torch.no_grad()
    def sample(self, x=None, c=0., **kwargs):
        if x is None:
            x = torch.randn((1, self.config["model_dim"]), device=self.device)
        x_shape = x.shape
        x = x.view(-1, x.size(-1))
        return self.diffusion_sampler(x, c, **kwargs).view(x_shape)

    @property
    def device(self):
        return next(self.parameters()).device




class OneDimVAE(nn.Module):
    def __init__(self, d_model, d_latent, last_length, kernel_size=7):
        super(OneDimVAE, self).__init__()
        self.d_model = d_model.copy()
        self.d_latent = d_latent
        self.last_length = last_length

        # Build Encoder
        modules = []
        in_dim = 1
        for h_dim in d_model:
            modules.append(nn.Sequential(
                nn.Conv1d(in_dim, h_dim, kernel_size, 2, kernel_size//2),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU()
            ))
            in_dim = h_dim
        self.encoder = nn.Sequential(*modules)
        self.to_latent = nn.Linear(self.last_length * d_model[-1], d_latent)
        self.fc_mu = nn.Linear(d_latent, d_latent)
        self.fc_var = nn.Linear(d_latent, d_latent)

        # Build Decoder
        modules = []
        self.to_decode = nn.Linear(d_latent, self.last_length * d_model[-1])
        d_model.reverse()
        for i in range(len(d_model) - 1):
            modules.append(nn.Sequential(
                nn.ConvTranspose1d(d_model[i], d_model[i+1], kernel_size, 2, kernel_size//2, output_padding=1),
                nn.BatchNorm1d(d_model[i + 1]),
                nn.ELU(),
            ))
        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(
            nn.ConvTranspose1d(d_model[-1], d_model[-1], kernel_size, 2, kernel_size//2, output_padding=1),
            nn.BatchNorm1d(d_model[-1]),
            nn.ELU(),
            nn.Conv1d(d_model[-1], 1, kernel_size, 1, kernel_size//2),
        )

    def encode(self, input, **kwargs):
        #print(input.shape)
        # assert input.shape == [batch_size, num_parameters]
        input = input[:, None, :]
        result = self.encoder(input)
        #print(result.shape)
        result = torch.flatten(result, start_dim=1)
        result = self.to_latent(result)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return mu, log_var

    def decode(self, z, **kwargs):
        # z.shape == [batch_size, d_latent]
        result = self.to_decode(z)
        result = result.view(-1, self.d_model[-1], self.last_length)
        result = self.decoder(result)
        result = self.final_layer(result)
        assert result.shape[1] == 1, f"{result.shape}"
        return result[:, 0, :]

    def reparameterize(self, mu, log_var, **kwargs):
        if kwargs.get("use_var"):
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            if kwargs.get("manual_std") is not None:
                std = kwargs.get("manual_std")
            return eps * std + mu
        else:  # not use var
            return mu

    def encode_decode(self, input, **kwargs):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var, **kwargs)
        recons = self.decode(z)
        return recons, input, mu, log_var

    def forward(self, x, **kwargs):
        recons, input, mu, log_var = self.encode_decode(input=x, **kwargs)
        recons_loss = F.mse_loss(recons, input)
        if kwargs.get("use_var"):
            kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
            loss = recons_loss + kwargs['kld_weight'] * kld_loss
        else:  # not use var
            loss = recons_loss
        return loss
