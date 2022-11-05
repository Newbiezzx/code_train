import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class Generator(nn.Module):
    def __init__(self, nz:int, ngf:int, nc:int) -> None:
        super().__init__()
        self.main = nn.Sequential(
            Generator._block(nz, ngf*8, 4, 1, 0),
            Generator._block(ngf*8, ngf*4, 4, 2, 1),
            Generator._block(ngf*4, ngf*2, 4, 2, 1),
            Generator._block(ngf*2, ngf, 4, 2, 1),
            nn.ConvTranspose2d(ngf, nc, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(nc),
            nn.Tanh()
        )
    
    def forward(self, input):
        return self.main(input)
    
    @staticmethod
    def _block(inch:int, outch:int, kernel_size=4, stride=1, padding=0, bias=False):
        return nn.Sequential(
            nn.ConvTranspose2d(inch, outch, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(outch),
            nn.ReLU(True)
        )

class Discriminator(nn.Module):
    def __init__(self, nc:int, ndf:int) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf*8, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )
    def forward(self, input):
        out = self.main(input)
        return out.view(-1, 1).squeeze(1)

def main():
    # model = Generator(100, 128, 3).cuda()
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = 'cpu'
    model = Discriminator(3, 128).to(device)
    summary(model, (3, 64, 64))

if __name__ == "__main__":
    main()