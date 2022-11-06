import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import argparse
import matplotlib.pyplot as plt

from model import VAE

# 设置超参数
parser = argparse.ArgumentParser(description='set VAE training hyperparameters')

parser.add_argument('--batch_size', '-b', default=64, type=int, help="batch size")
parser.add_argument('--lr', default=0.001, type=float, help="learning rate")
parser.add_argument('--epoch_max', default=10, type=int, help="epoch_max")

args = parser.parse_args()



# 训练函数
def train(model, trainloader, optim, epoch_max:int, draw=False):
    for epoch in range(epoch_max):
        epoch_loss = 0
        for _, (data, _) in enumerate(trainloader):
            data = data.cuda()
            optim.zero_grad()
            x_recon, mu, logvar = model(data)
            loss = loss_fuc(x_recon, data, mu, logvar)
            loss.backward()
            optim.step()

            epoch_loss += loss.item() 
        print(f"epoch:{epoch}, loss:{epoch_loss:.4f}")
    torch.save(model, "VAE/VAE_model.pth")
    if draw:
        batch_x_recon = make_grid(x_recon.detach().cpu().view(x_recon.size(0), 1, 28, 28), nrow=x_recon.size(0), padding=0)
        batch_x_real = make_grid(data.detach().cpu().view(data.size(0), 1, 28, 28), nrow=data.size(0), padding=0)

        batch_x_recon = np.transpose(batch_x_recon, (1,2,0))
        batch_x_real = np.transpose(batch_x_real, (1,2,0))
        plt.subplot(2,1,1)
        plt.imshow(batch_x_recon)
        plt.ylim((0, 28))
        plt.axis('off')
        plt.subplot(2,1,2)
        plt.imshow(batch_x_real)
        plt.ylim((0, 28))
        plt.axis('off')
        plt.subplots_adjust(left=None,bottom=None,right=None,top=None,wspace=0,hspace=0)
        plt.show()
            



# 损失函数
def loss_fuc(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(x.size(0), -1), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

if __name__ == "__main__":
    # 设置训练数据，模型，以及优化器
    datatrain = MNIST('data', train=True, download=True, transform=
        transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor()
        ])
    )

    trainloader = DataLoader(datatrain, batch_size=args.batch_size, shuffle=True)
    model = VAE().cuda()
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    epoch_max = args.epoch_max
    train(model, trainloader, optim, epoch_max, draw=True)
    