import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST,CIFAR10
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import argparse

from model import Generator, Discriminator

# 设置超参数
parser = argparse.ArgumentParser(description='set DCGAN training hyperparameters')

parser.add_argument('--batch_size', '-b', default=64, type=int, help="batch size")
parser.add_argument('--lr', default=0.001, type=float, help="learning rate")
parser.add_argument('--epoch_max', default=10, type=int, help="epoch_max")
parser.add_argument('--ngf', default=128, type=int)
parser.add_argument('--ndf', default=128, type=int)
parser.add_argument('--nc', default=3, type=int)
parser.add_argument('--nz', default=100, type=int)
parser.add_argument('--beta1', default=0.1, type=float)
parser.add_argument('--mean', default=[0.5, 0.5, 0.5], type=list)
parser.add_argument('--std', default=[0.5, 0.5, 0.5], type=list)

args = parser.parse_args()

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # 设置训练数据，模型，以及优化器
    datatrain = CIFAR10('data/CIFAR10', train=True, download=True, transform=
        transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean = args.mean,
                std = args.std
            )
        ])
    )

    trainloader = DataLoader(datatrain, batch_size=args.batch_size, shuffle=True)
    netG = Generator(nz=args.nz, ngf=args.ngf, nc=args.nc).to(device)
    netD = Discriminator(nc=args.nc, ndf=args.ndf).to(device)
    criterion = nn.BCELoss()
    optimG = torch.optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimD = torch.optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    epoch_max = args.epoch_max
    
    real_label = 1 # 真实图像标签
    fake_label = 0 # 生成图像标签

    for epoch in range(epoch_max):
        loss_D = 0
        loss_G = 0
        for (data, _) in trainloader:
            netD.zero_grad()
            real_cpu = data.to(device)
            batch_size = real_cpu.size(0)
            # 定义输入数据
            label = torch.full((batch_size,), real_label, device=device, dtype=torch.float32)
            output = netD(real_cpu)

            # 定义判别器相对于真实图像损失函数
            errD_real = criterion(output, label)
            # 梯度反向传播，相对于真实图像
            errD_real.backward()

            noise = torch.randn(batch_size, args.nz, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach())

            # 定义判别器相对于生成图像的损失函数
            errD_fake = criterion(output, label)
            errD_fake.backward()

            # 计算判别器总的损失函数，真实图像的损失函数+生成图像的损失函数
            errD = errD_fake + errD_real
            loss_D += errD.item()
            # 优化判别器
            optimD.step()

            # 生成器梯度置为0
            netG.zero_grad()
            label.fill_(real_label)
            output = netD(fake)
            # 定义判别器相对于真实图像损失函数
            errG = criterion(output, label)
            errG.backward()
            loss_G += errG.item()
            optimG.step()
        print(f"epoch:{epoch} | loss_G:{loss_G:.4f} | loss_D:{loss_D:.4f}")   
        torch.save(netG, f'DCGAN/model_save/Generator_model_{epoch}.pth')
        torch.save(netG, 'DCGAN/model_save/Generator_model_best.pth')