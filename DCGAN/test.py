import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import argparse

# 设置超参数
parser = argparse.ArgumentParser(description='set VAE training hyperparameters')

parser.add_argument('--batch_size', '-b', default=64, type=int, help="batch size")
parser.add_argument('--mean', default=[0.5, 0.5, 0.5], type=list)
parser.add_argument('--std', default=[0.5, 0.5, 0.5], type=list)

args = parser.parse_args()

def non_normalize(images_tensor, mean:list, std:list):
    """
    @description  :
    反归一化
    @param  :
    images_tensor:  批图片
    mean:  dataLoader中设置的mean参数
    std:   dataLoader中设置的std参数
    @Returns  :
    反归一化后的tensor
    """
    images_tensor = images_tensor.detach().cpu()
    batch = images_tensor.shape[0]
    std = torch.tensor(mean*batch).reshape(batch, len(std))
    mean = torch.tensor(mean*batch).reshape(batch, len(mean))
    mean_tensor = torch.einsum('bchw, bc->bchw', torch.ones_like(images_tensor), mean)
    output = torch.einsum('bchw, bc->bchw', images_tensor, std) + mean_tensor
    return output

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    model = torch.load('DCGAN/model_save/Generator_model_best.pth')
    input = torch.randn(args.batch_size, 100, 1, 1, device=device)
    output = model(input)
    output = non_normalize(output, args.mean, args.std)
    output = make_grid(output, padding=0)
    output = np.transpose(output, (1,2,0))
    plt.imshow(output)
    plt.show()

    