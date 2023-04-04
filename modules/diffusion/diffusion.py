import torch
import math
from torch import nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from functools import partial
from tqdm import tqdm

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def noise_like(shape, device, repeat=False):
    def repeat_noise(): return torch.randn(
        (1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))

    def noise(): return torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        image_size=256,
        channels=3,
        loss_type='l1',
        conditional=True,
        beta_schedule='linear',
        timesteps=2000,
        device='cuda'
    ):
        super().__init__()
        self.denoise_fn = denoise_fn

        self.image_size = image_size
        self.channels = channels
        self.loss_type = loss_type
        self.conditional = conditional
        if self.loss_type == 'l1':
            self.loss_func = nn.L1Loss(reduction='sum')
        elif self.loss_type == 'l2':
            self.loss_func = nn.MSELoss(reduction='sum')
        
        self.num_timesteps = int(timesteps)

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # equal to sqrt((1 - x_t / x_{t - 1}) * (1 - x_{t - 1}) / (x_t)) in ddim
        register_buffer('ddim_c1', torch.sqrt((1. - alphas_cumprod / alphas_cumprod_prev) * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)))

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(
            self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    @torch.no_grad()
    def generalized_steps(self, x_in, conditional_input=None, continous=False, noise_list=None ):
        device = self.betas.device
        sample_inter = (1 | (self.num_timesteps//10))

        b = x_in.size(0)
        img = x_in[:, :, :, :]
        ret_img = img
        for i in reversed(range(0, self.num_timesteps)):
            t = torch.full( (b,), i, device=device, dtype=torch.long)
            at = extract(self.alphas_cumprod, t, img.shape)
            at_next = extract(self.alphas_cumprod_prev, t, img.shape)
            et, _ = self.denoise_fn(img, t, conditional_input)
            x0_t = (img - et * (1 - at).sqrt()) / at.sqrt()
            c1 =1. * extract(self.ddim_c1, t, img.shape)
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            img = at_next.sqrt() * x0_t + c1 * noise_list[i] + c2 * et
            if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)
        if continous:
            return ret_img
        else:
            return ret_img[-1]
    
    @torch.no_grad()
    def ddpm_generalized_steps(self, x_in, conditional_input=None, continous=False, noise_list=None ):
        device = self.betas.device
        sample_inter = (1 | (self.num_timesteps//10))
        b = x_in.size(0)
        img = x_in[:, :, :, :]
        ret_img = img

        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            t = torch.full( (b,), i, device=device, dtype=torch.long)
            noise, _ = self.denoise_fn(img, t, conditional_input)
            x0_from_e = self.predict_start_from_noise(img, t=t, noise=noise)
            x0_from_e = torch.clamp(x0_from_e, -1, 1)
            model_mean, posterior_variance, model_log_variance = self.q_posterior(
                                                                     x_start=x0_from_e, x_t=img, t=t)
            nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(img.shape) - 1)))
            img = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise_list[i]
            if i % sample_inter == 0:
                ret_img = torch.cat([ret_img, img], dim=0)
        if continous:
            return ret_img
        else:
            return ret_img[-1]
    
    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        # fix gama
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod,
                    t, x_start.shape) * noise
        )

    def p_losses(self, x_in, noise=None):
        x_start = x_in['real']
        [b, c, h, w] = x_start.shape
        t = torch.randint(0, self.num_timesteps, (b,),
                          device=x_start.device).long()

        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        # condition is a dict with keys 'warping', and 'generated'        
        condition = x_in['condition']

        # here need to estimate pose from noise
        x_recon, loss_color = self.denoise_fn(x_noisy, t, condition)

        loss_dict = {}

        loss_dict['norm'] = self.loss_func(noise, x_recon)
        loss_dict['color'] = loss_color

        return loss_dict

    def forward(self, x, *args, **kwargs):
        return self.p_losses(x, *args, **kwargs)

