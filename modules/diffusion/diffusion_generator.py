from .diffusion import GaussianDiffusion
from .core_net import UNet
import torch
import torch.nn as nn


class DiffusionGenerator(nn.Module):
    def __init__(self, img_size=256, timesteps=100, condition=True, device='cuda'):
        super().__init__()
        core_model = UNet()
        self.diffusion = GaussianDiffusion(core_model, image_size=img_size, timesteps=timesteps, conditional=condition, device=device)
    
    def forward(self, x_in):
        return self.diffusion(x_in)

    @torch.no_grad()
    def refer(self, x_in, condition=None, noise_list=None):
        return self.diffusion.generalized_steps(x_in, conditional_input=condition, continous=True, noise_list=noise_list)

