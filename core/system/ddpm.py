import pdb

import hydra.utils
import pytorch_lightning as pl
import torch
from typing import Any
import numpy as np
import torch.nn as nn

from .base import BaseSystem
from core.utils.ddpm import *
from core.utils.utils import *
from core.module.prelayer.latent_transformer import Param2Latent
from .encoder import EncoderSystem

class DDPM(BaseSystem):
    def __init__(self, config, **kwargs):
        super(DDPM, self).__init__(config)
        self.save_hyperparameters()
        betas = self.config.beta_schedule
        self.n_timestep = betas.n_timestep
        betas = make_beta_schedule(**betas)
        self.betas_register(betas)

    def betas_register(self, betas):
        model_mean_type = self.config.model_mean_type
        model_var_type = self.config.model_var_type
        betas = betas.type(torch.float64)
        timesteps = betas.shape[0]
        self.num_timesteps = int(timesteps)

        self.model_mean_type = model_mean_type  # xprev, xstart, eps
        self.model_var_type = model_var_type  # learned, fixedsmall, fixedlarge
        self.loss_type = self.config.loss_type  # kl, mse

        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, 0)
        alphas_cumprod_prev = torch.cat(
            (torch.tensor([1], dtype=torch.float64), alphas_cumprod[:-1]), 0
        )
        posterior_variance = betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)

        self.register("betas", betas)
        self.register("alphas_cumprod", alphas_cumprod)
        self.register("alphas_cumprod_prev", alphas_cumprod_prev)

        self.register("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register("sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - alphas_cumprod))
        self.register("log_one_minus_alphas_cumprod", torch.log(1 - alphas_cumprod))
        self.register("sqrt_recip_alphas_cumprod", torch.rsqrt(alphas_cumprod))
        self.register("sqrt_recipm1_alphas_cumprod", torch.sqrt(1 / alphas_cumprod - 1))
        self.register("posterior_variance", posterior_variance)
        self.register("posterior_log_variance_clipped",
                      torch.log(torch.cat((posterior_variance[1].view(1, 1),
                                           posterior_variance[1:].view(-1, 1)), 0)).view(-1))
        self.register("posterior_mean_coef1", (betas * torch.sqrt(alphas_cumprod_prev) / (1 - alphas_cumprod)))
        self.register("posterior_mean_coef2", ((1 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1 - alphas_cumprod)))

    def generate(self, batch, num=10, history=False):
        model = self.model.ema if hasattr(self.model, 'ema') else self.model
        model.eval()
        shape = (num, 1, batch.shape[1] * batch.shape[2])
        sample = self.progressive_samples_fn_simple(
            model,
            shape,
            device='cuda',
            # cond = cond_input_val,
            include_x0_pred_freq=50,
            history=history,
        )

        if history:
            return sample['samples'], sample['history']
        return sample['samples']


    def progressive_samples_fn_simple(self, model, shape, device, include_x0_pred_freq=50, history=False):
        samples, history = self.p_sample_loop_progressive_simple(
            model=model,
            shape=shape,
            noise_fn=torch.randn,
            device=device,
            include_x0_pred_freq=include_x0_pred_freq,
            # cond=cond,
        )
        if history:
            return {'samples': samples, 'history': history}
        return {'samples': samples}

    def pre_process(self, batch):
        if hasattr(self, 'data_transform') and self.data_transform is not None:
            batch = self.data_transform.pre_process(batch)
        return batch

    def post_process(self, outputs):
        if hasattr(self, 'data_transform') and self.data_transform is not None:
            outputs = self.data_transform.post_process(outputs)
        return outputs

    def validation_step(self, batch, batch_idx, **kwargs: Any):
        batch = self.pre_process(batch)
        outputs = self.generate(batch, 10)

        params = self.post_process(outputs)
        params = params.cpu()

        accs = []
        for i in range(params.shape[0]):
            param = params[i].to(batch.device)
            acc, test_loss, output_list = self.task_func(param)
            accs.append(acc)
        best_acc = np.max(accs)
        print("generated models accuracy:", accs)
        print("generated models mean accuracy:", np.mean(accs))
        print("generated models best accuracy:", best_acc)
        self.log('best_g_acc', best_acc)
        self.log('mean_g_acc', np.mean(accs).item())
        return {'best_g_acc': best_acc, 'mean_g_acc': np.mean(accs).item()}

    def test_step(self, batch, batch_idx, **kwargs: Any):
        # generate 100 models
        batch = self.pre_process(batch)
        outputs = self.generate(batch, 100)
        params = self.post_process(outputs)
        accs = []
        for i in range(params.shape[0]):
            param = params[i]
            acc, test_loss, output_list = self.task_func(param)
            accs.append(acc)
        best_acc = np.max(accs)
        print("generated models accuracy:", accs)
        print("generated models mean accuracy:", np.mean(accs))
        print("generated models best accuracy:", best_acc)
        print("generated models median accuracy:", np.median(accs))
        self.log('best_g_acc', best_acc)
        self.log('mean_g_acc', np.mean(accs).item())
        self.log('med_g_acc', np.median(accs).item())
        return {'best_g_acc': best_acc, 'mean_g_acc': np.mean(accs).item(), 'med_g_acc': np.median(accs).item()}

    def forward(self, batch, **kwargs):
        batch = self.pre_process(batch)
        model = self.model
        time = (torch.rand(batch.shape[0]) * self.n_timestep).type(torch.int64).to(batch.device)

        noise = None
        lab = None
        if noise is None:
            noise = torch.randn_like(batch)
        x_t = self.q_sample(batch, time, noise=noise)

        # todo: loss using criterion, so we can change it
        if self.loss_type == 'kl':
            # the variational bound
            losses = self._vb_terms_bpd(model=model, x_0=batch, x_t=x_t, t=time, clip_denoised=False, return_pred_x0=False)

        elif self.loss_type == 'mse':
            # unweighted MSE
            assert self.model_var_type != 'learned'
            target = {
                'xprev': self.q_posterior_mean_variance(x_0=batch, x_t=x_t, t=time)[0],
                'xstart': batch,
                'eps': noise
            }[self.model_mean_type]

            model_output = model(x_t, time, cond=lab)
            losses       = torch.mean((target - model_output).view(batch.shape[0], -1)**2, dim=1)

        else:
            raise NotImplementedError(self.loss_type)

        loss = losses.mean()

        # todo: ema is a insert
        if hasattr(self.model, 'ema'):
            accumulate(self.model.ema,
                       self.model.model if isinstance(self.model.model, nn.DataParallel) else self.model.model, 0.9999)

        self.log('train_loss', loss)
        return loss


    def register(self, name, tensor):
        self.register_buffer(name, tensor.type(torch.float32))

    def _prior_bpd(self, x_0):

        B, T                        = x_0.shape[0], self.num_timesteps
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_0,
                                                           t=torch.full((B,), T - 1, dtype=torch.int64))
        kl_prior                    = normal_kl(mean1=qt_mean,
                                                logvar1=qt_log_variance,
                                                mean2=torch.zeros_like(qt_mean),
                                                logvar2=torch.zeros_like(qt_log_variance))

        return torch.mean(kl_prior.view(B, -1), dim=1)/np.log(2.)

    @torch.no_grad()
    def calc_bpd_loop(self, model, x_0, clip_denoised):

        (B, C, H, W), T = x_0.shape, self.num_timesteps

        new_vals_bt = torch.zeros((B, T))
        new_mse_bt  = torch.zeros((B, T))

        for t in reversed(range(self.num_timesteps)):

            t_b = torch.full((B, ), t, dtype=torch.int64)

            # Calculate VLB term at the current timestep
            new_vals_b, pred_x0 = self._vb_terms_bpd(model=model,
                                                     x_0=x_0,
                                                     x_t=self.q_sample(x_0=x_0, t=t_b),
                                                     t=t_b,
                                                     clip_denoised=clip_denoised,
                                                     return_pred_x0=True)

            # MSE for progressive prediction loss
            new_mse_b = torch.mean((pred_x0-x_0).view(B, -1)**2, dim=1)

            # Insert the calculated term into the tensor of all terms
            mask_bt = (t_b[:, None] == torch.arange(T)[None, :]).to(torch.float32)

            new_vals_bt = new_vals_bt * (1. - mask_bt) + new_vals_b[:, None] * mask_bt
            new_mse_bt  = new_mse_bt  * (1. - mask_bt) + new_mse_b[:, None] * mask_bt

        prior_bpd_b = self._prior_bpd(x_0)
        total_bpd_b = torch.sum(new_vals_bt, dim=1) + prior_bpd_b

        return total_bpd_b, new_vals_bt, prior_bpd_b, new_mse_bt

    def q_mean_variance(self, x_0, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_0.shape) * x_0
        variance = extract(1. - self.alphas_cumprod, t, x_0.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_0.shape)
        return mean, variance, log_variance

    def q_sample(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)

        return (extract(self.sqrt_alphas_cumprod, t, x_0.shape) * x_0
                + extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape) * noise)


    def q_posterior_mean_variance(self, x_0, x_t, t):
        mean            = (extract(self.posterior_mean_coef1, t, x_t.shape) * x_0
                           + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t)
        var             = extract(self.posterior_variance, t, x_t.shape)
        log_var_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)

        return mean, var, log_var_clipped

    # def p_mean_variance(self, model, x, t, clip_denoised, return_pred_x0, lab):
    def p_mean_variance(self, model, x, t, clip_denoised, return_pred_x0):

        # import pdb; pdb.set_trace()
        # model_output = model(x, t, cond=lab)
        model_output = model(x, t)


        # Learned or fixed variance?
        if self.model_var_type == 'learned':
            model_output, log_var = torch.split(model_output, 2, dim=-1)
            var                   = torch.exp(log_var)

        elif self.model_var_type in ['fixedsmall', 'fixedlarge']:

            # below: only log_variance is used in the KL computations
            var, log_var = {
                # for 'fixedlarge', we set the initial (log-)variance like so to get a better decoder log likelihood
                'fixedlarge': (self.betas, torch.log(torch.cat((self.posterior_variance[1].view(1, 1),
                                                                self.betas[1:].view(-1, 1)), 0)).view(-1)),
                'fixedsmall': (self.posterior_variance, self.posterior_log_variance_clipped),
            }[self.model_var_type]
            # import pdb;pdb.set_trace()

            var     = extract(var, t, x.shape) * torch.ones_like(x)
            log_var = extract(log_var, t, x.shape) * torch.ones_like(x)
        else:
            raise NotImplementedError(self.model_var_type)

        # Mean parameterization
        _maybe_clip = lambda x_: (x_.clamp(min=-1, max=1) if clip_denoised else x_)

        if self.model_mean_type == 'xprev':
            # the model predicts x_{t-1}
            pred_x_0 = _maybe_clip(self.predict_start_from_prev(x_t=x, t=t, x_prev=model_output))
            mean     = model_output
        elif self.model_mean_type == 'xstart':
            # the model predicts x_0
            pred_x0    = _maybe_clip(model_output)
            mean, _, _ = self.q_posterior_mean_variance(x_0=pred_x0, x_t=x, t=t)
        elif self.model_mean_type == 'eps':
            # the model predicts epsilon
            pred_x0    = _maybe_clip(self.predict_start_from_noise(x_t=x, t=t, noise=model_output))
            mean, _, _ = self.q_posterior_mean_variance(x_0=pred_x0, x_t=x, t=t)
        else:
            raise NotImplementedError(self.model_mean_type)

        if return_pred_x0:
            return mean, var, log_var, pred_x0
        else:
            return mean, var, log_var

    def predict_start_from_noise(self, x_t, t, noise):
        # import pdb; pdb.set_trace()

        return (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise)

    def predict_start_from_prev(self, x_t, t, x_prev):

        return (extract(1./self.posterior_mean_coef1, t, x_t.shape) * x_prev -
                extract(self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape) * x_t)

    # def p_sample(self, model, x, t, noise_fn, clip_denoised=True, return_pred_x0=False, lab=None):
    def p_sample(self, model, x, t, noise_fn, clip_denoised=True, return_pred_x0=False):
        # pdb.set_trace()
        mean, _, log_var, pred_x0 = self.p_mean_variance(model, x, t, clip_denoised, return_pred_x0=True)

        noise                     = noise_fn(x.shape, dtype=x.dtype).to(x.device)

        shape        = [x.shape[0]] + [1] * (x.ndim - 1)
        nonzero_mask = (1 - (t == 0).type(torch.float32)).view(*shape).to(x.device)
        sample       = mean + nonzero_mask * torch.exp(0.5 * log_var) * noise

        sample = torch.clamp(sample, min=-1, max=1)

        # self.monitor.wandb_log({f'k60_noise_norm/{batch}': ((torch.exp(0.5 * log_var) * noise).norm().item())})
        # self.monitor.wandb_log({f'k60_noise_image/{batch}': wandb.Image(torch.exp(0.5 * log_var)* noise)})

        return (sample, pred_x0) if return_pred_x0 else sample

    @torch.no_grad()
    # def p_sample_loop(self, model, shape, noise_fn=torch.randn, lab=None):
    def p_sample_loop(self, model, shape, noise_fn=torch.randn):


        device = 'cuda' if next(model.parameters()).is_cuda else 'cpu'
        # shape[0] = lab.shape[0]
        img    = noise_fn(10).to(device)

        for i in reversed(range(self.num_timesteps)):
            img = self.p_sample(
                model,
                img,
                torch.full((shape[0],), i, dtype=torch.int64).to(device),
                noise_fn=noise_fn,
                return_pred_x0=False,
                # lab=lab,
            )

        return img

    @torch.no_grad()
    def p_sample_loop_progressive(self, model, shape, device, cond, noise_fn=torch.randn, include_x0_pred_freq=50):

        img = noise_fn(shape, dtype=torch.float32).to(device)
        num_recorded_x0_pred = self.num_timesteps // include_x0_pred_freq
        x0_preds_            = torch.zeros((shape[0], num_recorded_x0_pred, *shape[1:]), dtype=torch.float32).to(device)

        for i in reversed(range(self.num_timesteps)):

            img, pred_x0 = self.p_sample(model=model,
                                         x=img,
                                         t=torch.full((shape[0],), i, dtype=torch.int64).to(device),
                                         noise_fn=noise_fn,
                                         return_pred_x0=True,
                                        #  lab=cond,
                                         )

            # Keep track of prediction of x0
            insert_mask = np.floor(i // include_x0_pred_freq) == torch.arange(num_recorded_x0_pred,
                                                                              dtype=torch.int32,
                                                                              device=device)

            insert_mask = insert_mask.to(torch.float32).view(1, num_recorded_x0_pred, *([1] * len(shape[1:])))
            x0_preds_   = insert_mask * pred_x0[:, None, ...] + (1. - insert_mask) * x0_preds_

        return img, x0_preds_

    def p_sample_loop_progressive_simple(self, model, shape, device, noise_fn=torch.randn,
                                         include_x0_pred_freq=50,input_pa=None,exp_step=None):

        # import pdb; pdb.set_trace()
        # sample_lab = torch.tensor([i%10 w i in range(shape[0])]).long().to(device)
        img = noise_fn(shape, dtype=torch.float32).to(device)
        if input_pa is not None:
            img = input_pa.repeat(shape[0],shape[1],1).to(device)
        num_recorded_x0_pred = self.num_timesteps // include_x0_pred_freq
        x0_preds_            = torch.zeros((shape[0], num_recorded_x0_pred, *shape[1:]), dtype=torch.float32).to(device)

        history = []
        if exp_step is not None:
            step = exp_step
        else:
            step = self.num_timesteps
        for i in reversed(range(step)):
            # import pdb;pdb.set_trace()

            # import pdb; pdb.set_trace()
            # Sample p(x_{t-1} | x_t) as usual
            img, pred_x0 = self.p_sample(model=model,
                                         x=img,
                                         t=torch.full((shape[0],), i, dtype=torch.int64).to(device),
                                         noise_fn=noise_fn,
                                         return_pred_x0=True,
                                        #  lab=cond,
                                         )


            history.append(img.detach().cpu())
        return img, history

    # === Log likelihood calculation ===

    def _vb_terms_bpd(self, model, x_0, x_t, t, clip_denoised, return_pred_x0, lab=None):

        batch_size = t.shape[0]
        true_mean, _, true_log_variance_clipped    = self.q_posterior_mean_variance(x_0=x_0,
                                                                                    x_t=x_t,
                                                                                    t=t)
        model_mean, _, model_log_variance, pred_x0 = self.p_mean_variance(model,
                                                                          x=x_t,
                                                                          t=t,
                                                                          clip_denoised=clip_denoised,
                                                                          return_pred_x0=True,
                                                                          lab=lab,
                                                                          )

        kl = normal_kl(true_mean, true_log_variance_clipped, model_mean, model_log_variance)
        kl = torch.mean(kl.view(batch_size, -1), dim=1) / np.log(2.)

        decoder_nll = -discretized_gaussian_log_likelihood(x_0, means=model_mean, log_scales=0.5 * model_log_variance)
        decoder_nll = torch.mean(decoder_nll.view(batch_size, -1), dim=1) / np.log(2.)

        # At the first timestep return the decoder NLL, otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = torch.where(t == 0, decoder_nll, kl)

        return (output, pred_x0) if return_pred_x0 else output


