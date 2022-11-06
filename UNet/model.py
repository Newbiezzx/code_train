import torch
import torch.nn as nn
from torchsummary import summary

class UNet(nn.Module):
    def __init__(self, inch, outch, init_feature=64) -> None:
        super().__init__()

        features = init_feature
        # 下采样
        channels = [2**i*features for i in range(5)]
        self.encode1 = UNet._block(inch, channels[0])
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encode2 = UNet._block(channels[0], channels[1])
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encode3 = UNet._block(channels[1], channels[2])
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encode4 = UNet._block(channels[2], channels[3])
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(channels[3], channels[4])
        # 上采样
        self.upconv4 = nn.ConvTranspose2d(channels[4], channels[3], kernel_size=2, stride=2) 
        self.dec4 = UNet._block(channels[3]*2, channels[3])

        self.upconv3 = nn.ConvTranspose2d(channels[3], channels[2], kernel_size=2, stride=2) 
        self.dec3 = UNet._block(channels[2]*2, channels[2])

        self.upconv2 = nn.ConvTranspose2d(channels[2], channels[1], kernel_size=2, stride=2) 
        self.dec2 = UNet._block(channels[1]*2, channels[1])
         
        self.upconv1 = nn.ConvTranspose2d(channels[1], channels[0], kernel_size=2, stride=2) 
        self.dec1 = UNet._block(channels[0]*2, channels[0])

        self.conv = nn.Conv2d(channels[0], outch, kernel_size=1)

    def forward(self, x):
        # 下采样过程
        enc1 = self.encode1(x)
        enc2 = self.encode2(self.maxpool1(enc1))
        enc3 = self.encode3(self.maxpool2(enc2))
        enc4 = self.encode4(self.maxpool3(enc3))
        bottleneck = self.bottleneck(self.maxpool4(enc4))

        # 上采样过程
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.dec4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)

        out = self.conv(dec1)
        return torch.sigmoid(out)

    @staticmethod
    def _block(inch:int, outch:int):
        return nn.Sequential(
            nn.Conv2d(inch, outch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(outch),
            nn.ReLU(inplace=True),

            nn.Conv2d(outch, outch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(outch),
            nn.ReLU(inplace=True)
        )

def main():
    model = UNet(3, 1).cuda()
    summary(model, (3, 256, 256), batch_size=64)

if __name__ == "__main__":
    main()