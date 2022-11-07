import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from copy import deepcopy

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
    def __init__(self, model:nn.Module, img_channels, img_size, \
        num_classes, betas, loss_type='l2', ema_decay=0.9999,\
        ema_start=5000, ema_update_rate=1) -> None:
        super().__init__()

        self.model = model
        self.ema_model = deepcopy(model)

        self.ema = EMA(ema_decay)

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


    
    




