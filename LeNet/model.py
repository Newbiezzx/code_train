import torch
import torch.nn as nn
import  torch.nn.functional as F
from torchsummary import summary


class LeNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(1, 6, 3),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Conv2d(6, 16, 3),
            nn.MaxPool2d(2, 2),
            nn.ReLU()
        )

        self.fc1 = nn.Sequential(
            nn.Linear(16*6*6, 120),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(84, 10)


    def forward(self, x):
        x = self.downsample(x)
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x

        
if __name__ == "__main__":
    model = LeNet().cuda()
    summary(model, (1, 32, 32), batch_size=100)

