import torch
import torch.nn as nn
from torchsummary import summary

class ResBlock(nn.Module):
    """
    @description  :
    残差模块
    """
    def __init__(self, inch, outch, stride=1, downsample=None, groups=1,
        bsae_width=64, dilation=1, normlayer=None) -> None:
        super().__init__()
        if normlayer is None:   
            normlayer = nn.BatchNorm2d
        
        self.branch0 = nn.Sequential(
            nn.Conv2d(inch, inch*2, 3, stride, padding=dilation, groups=groups, bias=False, dilation=dilation),
            normlayer(inch*2),
            nn.ReLU(),

            nn.Conv2d(inch*2, outch, 3, stride, padding=dilation, groups=groups, bias=False, dilation=dilation),
            normlayer(outch)
        )
        self.downsample = downsample
        
    def forward(self, x):
        identity = x
        out = self.branch0(x)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = torch.relu(out)
        return out

class Resnet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        




model = ResBlock(3, 3).cuda()
print(summary(model, (3, 512, 512), batch_size=1))