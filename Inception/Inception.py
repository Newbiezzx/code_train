import torch.nn as nn
import torch
from torchsummary import summary


class BasicConv2d(nn.Module):
    """
    卷积操作模块包含一个卷积层， 一个归一化层，一个激活函数层
    """
    def __init__(self, in_ch, out_ch, kernel_size=1, stride=1, padding=0) -> None:
        super().__init__()
        self.conv2D = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv2D(x)
        x = self.bn(x)
        y = self.relu(x)
        return y

class InceptionA(nn.Module):
    """
    Inception_A模型框架
    """
    def __init__(self, in_ch, out_ch) -> None:
        super().__init__()
        self.branch0 = BasicConv2d(in_ch, out_ch, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(in_ch, 32, kernel_size=1),
            BasicConv2d(32, out_ch, kernel_size=5, padding=2)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_ch, 32, kernel_size=1),
            BasicConv2d(32, 64, kernel_size=3, padding=1),
            BasicConv2d(64, out_ch, kernel_size=3, padding=1)
        )
        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),   
            # count_include_pad为True时，padding补的0需要在计算时加入.
            
            BasicConv2d(in_ch, out_ch, kernel_size=1)
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out
    

class InceptionB(nn.Module):
    """
    @description  :
    InceptionB模块，包含三个分支
    分支1：一个3*3的卷积
    分支2：一个1*1的卷积，两个3*3的卷积
    分支3：一个3*3的最大池化
    ---------
    @param  :
    in_ch:输入的通道数
    out_ch:输出通道数
    """
    
    def __init__(self, in_ch, out_ch) -> None:
        super().__init__()
        self.branch0 = BasicConv2d(in_ch, out_ch, kernel_size=3, stride=2) 
        self.branch1 = nn.Sequential(
            BasicConv2d(in_ch, 32, kernel_size=1),
            BasicConv2d(32, 64, kernel_size=3, padding=1),
            BasicConv2d(64, out_ch, kernel_size=3, stride=2)
        )
        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out

class InceptionC(nn.Module):
    """
    @description  :
    InceptionC模块，包含4个分支
    分支1：一个1*1的卷积
    分支2：一个1*1的卷积，一个1*7的卷积， 一个7*1的卷积
    分支3：一个1*1的卷积，一个1*7的卷积， 一个7*1的卷积， 一个1*7的卷积， 一个7*1的卷积
    分支4:一个3*3的平均池化， 一个1*1的卷积
    ---------
    @param  :
    in_ch:输入的通道数
    out_ch:输出通道数
    """
    def __init__(self, in_ch, out_ch) -> None:
        super().__init__()
        self.branch0 = BasicConv2d(in_ch, out_ch, kernel_size=1) 
        self.branch1 = nn.Sequential(
            BasicConv2d(in_ch, 32, kernel_size=1),
            BasicConv2d(32, 64, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(64, out_ch, kernel_size=(7, 1), padding=(3, 0))
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_ch, 32, kernel_size=1),
            BasicConv2d(32, 64, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(64, 32, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(32, 64, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(64, out_ch, kernel_size=(7, 1), padding=(3, 0))
        )
        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            BasicConv2d(in_ch, out_ch)
        )


    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out

class InceptionD(nn.Module):
    """
    @description  :
    InceptionD模块，包含三个分支
    分支1：一个1*1的卷积，一个3*3的卷积
    分支2：一个1*1的卷积，两个1*7的卷积，一个7*1的卷积， 一个3*3的卷积
    分支3：一个3*3的最大池化
    ---------
    @param  :
    in_ch:输入的通道数
    out_ch:输出通道数
    """
    def __init__(self, in_ch, out_ch) -> None:
        super().__init__()
        self.branch0 = nn.Sequential(
            BasicConv2d(in_ch, 32, kernel_size=1),
            BasicConv2d(32, out_ch, kernel_size=3, stride=2)
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_ch, 32, kernel_size=1),
            BasicConv2d(32, 64, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(64, 96, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(96, out_ch, kernel_size=3, stride=2)
        )
        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out

class InceptionE(nn.Module):
    """
    @description  :
    InceptionE模块，包含四个分支
    分支1：一个1*1的卷积
    分支2：一个1*1的卷积， 一个1*3的卷积和一个3*1的卷积并联
    分支3：一个1*1的卷积， 一个1*3的卷积和一个3*1的卷积并联， 一个3*3的卷积
    分支4：一个3*3的平均池化， 一个1*1的卷积
    ---------
    @param  :
    in_ch:输入的通道数
    out_ch:输出通道数
    """
    def __init__(self, in_ch, out_ch) -> None:
        super().__init__()
        self.branch0 = BasicConv2d(in_ch, out_ch)
        
        self.branch1_1 = BasicConv2d(in_ch, out_ch)
        self.branch1_2a = BasicConv2d(out_ch, out_ch, (3, 1), padding=(1,0))
        self.branch1_2b = BasicConv2d(out_ch, out_ch, (1, 3), padding=(0,1))
        
        self.branch2_1 = BasicConv2d(in_ch, 64)
        self.branch2_2a = BasicConv2d(64, 64, (3, 1), padding=(1,0)) 
        self.branch2_2b = BasicConv2d(64, 64, (1, 3), padding=(0,1))
        self.branch2_3 =  BasicConv2d(128, out_ch, 3, padding=1)


        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            BasicConv2d(in_ch, out_ch)
        )

    def forward(self, x):
        b0 = self.branch0(x)
        
        b1_1 = self.branch1_1(x)
        b1 = torch.cat((self.branch1_2a(b1_1), self.branch1_2b(b1_1)), 1)
        
        b2_1 = self.branch2_1(x)
        b2_2 = torch.cat((self.branch2_2a(b2_1), self.branch2_2b(b2_1)), 1)
        b2 = self.branch2_3(b2_2)

        b3 = self.branch3(x)
        return torch.cat((b0, b1, b2, b3), 1)

mymodel = InceptionE(3, 96).cuda()
print(summary(mymodel, (3, 256, 256), batch_size=1))


