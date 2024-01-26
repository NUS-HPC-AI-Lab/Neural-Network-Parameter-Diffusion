import torch
import torch.nn.functional as F
import torch.nn as nn

class mse(nn.MSELoss):
    def __init__(self):
        super(mse, self).__init__()


class VAELoss(nn.Module):
    def __init__(self):
        super(VAELoss, self).__init__()


    def forward(self, *args, **kwargs):
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset
        recons_loss = F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss.detach(), 'KLD': -kld_loss.detach()}