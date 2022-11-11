import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from copy import deepcopy

from utils import extract

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
    def __init__(self, model:nn.Module, img_channels:int, *img_size, \
        num_classes:int, betas:np.ndarray, loss_type='l2', ema_decay=0.9999,\
        ema_start=5000, ema_update_rate=1) -> None:
        super().__init__()

        self.model = model
        self.ema_model = deepcopy(model)
        self.img_channels = img_channels

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
        alphas_cumprod = np.cumprod(alphas) # 连乘函数\bar{alpha}
        to_torch = partial(torch.tensor, dtype=torch.float32) # 将函数转为tensor

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
    
    @torch.no_grad()
    def remove_noise(self, x, t:int, y, use_ema=True):
        if use_ema:
            return (
                (x - extract(self.remove_noise_coeff, t, x.shape) * self.ema_model(x, t, y)) *
                extract(self.reciprocal_sqrt_alphas, t, x.shape)
            )
        else:
            return (
                (x - extract(self.remove_noise_coeff, t, x.shape) * self.model(x, t, y)) *
                extract(self.reciprocal_sqrt_alphas, t, x.shape)
            )
    
    @torch.no_grad()
    def sample(self, batch_size:int, device:str, y=None, use_ema=True):
        if y is not None and batch_size != len(y):
            raise ValueError("sample batch size different from length of given y")
        
        x = torch.randn(batch_size, self.img_channels, *self.img_size, device=device)
        
        for t in range(self.num_timestep-1, -1, -1):
            t_batch = torch.tensor([t], device=device).repeat(batch_size)
            x = self.remove_noise(x, t_batch, y, use_ema)
        
            if t > 0:
                x += extract(self.sigma, t_batch, x.shape) * torch.randn_like(x)
        return x.cpu().detach()
    
    @torch.no_grad()
    def sample_diffusion_sequence(self, batch_size, device, y=None, use_ema=True):
        if y is not None and batch_size != len(y):
            raise ValueError("sample batch size different from length of given y")
        
        x = torch.randn(batch_size, self.img_channels, *self.img_size, device=device)
        diffusion_sequence = [x.cpu().detach]
        
        for t in range(self.num_timestep-1, -1, -1):
            t_batch = torch.tensor([t], device=device).repeat(batch_size)
            x = self.remove_noise(x, t_batch, y, use_ema)
        
            if t > 0:
                x += extract(self.sigma, t_batch, x.shape) * torch.randn_like(x)
            diffusion_sequence.append(x.cpu().detach())  
        return diffusion_sequence
    
    def preturb_x(self, x, t, noise):
        return(
            extract(self.sqrt_alphas_cumprod, t, x.shape) * x + 
            extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * noise
        )
        
    def get_losses(self, x, t, y):
        noise = torch.randn_like(x)
        preturb_x = self.preturb_x(x, t, noise)
        estimated_noise = self.model(preturb_x, t, y)
        if self.loss_type == "l1":
            loss = F.l1_loss(estimated_noise, noise)
        else:
            loss = F.mse_loss(estimated_noise, noise)
        return loss
    
    def forward(self, x, y=None):
        b, c, h, w = x.shape
        device = x.device
        
        if h != self.img_size[0]:
            raise ValueError("image height does not match diffusion parameters")
        if w != self.img_size[1]:
            raise ValueError("image width does not match diffusion parameters")
        
        t = torch.randint(0, self.num_timestep, (b,), device=device)
        return self.get_losses(x, t, y)
    
def generate_cosine_schedule(T:int, s=0.008):
    def f(t, T):
        return (np.cos((t / T + s) / (1 + s) * np.pi / 2)) ** 2
    
    alphas = []
    f0 = f(0, T)
    for t in range(T+1):
        alphas.append(f(t, T) / f0)
    betas = []
    
    for t in range(1, T+1):
        betas.append(min(1 - alphas[t] / alphas[t - 1], 0.999))
    
    return np.array(betas)

def generate_linear_schedule(T:int, low, high):
    return np.linspace(low, high, T)
        
        
  
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
    

if __name__ == "__main__":
    betas = generate_linear_schedule(100, 0.99, 0.9)
    model = GaussianDiffusion(nn.Module, 3, 32, 32, num_classes=10, betas=betas)
    y = model(torch.randn(1, 3, 32, 32))


