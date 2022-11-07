import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from copy import deepcopy

from .utils import extract

class GaussianDiffusion(nn.Module):
    """
    @description  :
    Gaussian Diffusion model. Forwarding through the 
    module returns diffusion reversal scalar loss tensor
    ---------
    @param  :
    x :tensor of shape (batch_size, img_channels, H, W)
    y :tensor of shape (batch_size)
    -------
    @Returns  :
    scalar loss tensor
    -------
    """
    def __init__(self, model:nn.Module, img_channels:int, img_size, \
        num_classes:int, betas:float, loss_type='l2', ema_decay=0.9999,\
        ema_start=5000, ema_update_rate=1) -> None:
        super().__init__()

        self.model = model
        self.ema_model = deepcopy(model)

        self.ema = EMA(ema_decay)
        self.ema_decay = ema_decay
        self.ema_start = ema_start
        self.ema_update_rate = ema_update_rate
        self.step = 0
        self.img_size = img_size
        self.num_classes = num_classes
        if loss_type not in ["l1", "l2"]:
            raise ValueError("__init__() got unknown loss type")
        else:
            self.loss_type = loss_type
        self.num_timestep = len(betas)

        alphas =  1 - betas
        alphas_cumprod = np.cumprod(alphas)
        to_torch = partial(torch.tensor, dytype=torch.float32)

        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas", to_torch(alphas))
        self.register_buffer("alpha_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1 - alphas_cumprod)))
        self.register_buffer("reciprocal_sqrt_alphas", to_torch(np.sqrt(1 / alphas)))
        self.register_buffer("remove_noise_coeff", to_torch(betas / np.sqrt(1 / alphas_cumprod)))
        self.register_buffer("sigma", to_torch(np.sqrt(betas)))
    
    def update_ema(self):
        self.step += 1
        if self.step % self.ema_update_rate == 0:
            if self.step < self.ema_start:
                self.ema_model.load_state_dict(self.model.state_dict())
            else:
                self.ema.update_model_average(self.ema_model, self.model)
    
    @torch.no_grad           
    def remove_noise(self, x, t, y, use_ema=True):
        if use_ema:
            return (
                (x - extract(self.remove_noise_coffe, t, x.shape) * self.ema_model(x, t, y)) *
                extract(self.reciprocal_sqrt_alphas, t, x_shape)
            )
        else:
            return (
                (x - extract(self.remove_noise_coffe, t, x.shape) * self.model(x, t, y)) *
                extract(self.reciprocal_sqrt_alphas, t, x_shape)
            )
    
    @torch.no_grad
    def sample(self, batch_size:int, devide:str, y=None, use_ema=True):
        if y is not None and batch_size != len(y):
            raise ValueError("sample batch size different from length of given y")
class EMA():
    def __init__(self, decay) -> None:
        self.decay = decay
    
    def updata_average(self, old, new):
        if old == None:
            return new
        return old * self.decay + (1 - self.decay) * new

    def update_model_average(self, ema_model:nn.Module, current_model:nn.Module):
        for current_params, ema_params in zip(current_model.parameters(), ema_model.parameters()):
            old, new = ema_params.data, current_params.data
            ema_params.data = self.updata_average(old, new)   
    




