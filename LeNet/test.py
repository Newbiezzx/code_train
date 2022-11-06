import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from torch.utils.data import DataLoader

from model import LeNet

datatest = MNIST("data", train=False, transform=(transforms.Compose(
        [transforms.Resize((32, 32)),
        transforms.ToTensor()]
    )), download=True)

test_loader = DataLoader(datatest, 100, shuffle=True)

model = LeNet().cuda()
model.eval()    # 将网络改为测试模式
# 加载模型参数
model.load_state_dict(torch.load("LeNet/LeNet_para_epoch9.pth"))
crit = nn.CrossEntropyLoss()
train_loss = 0
correct = 0
total = 0
for _, (data, target) in enumerate(test_loader):
    data, target = data.cuda(), target.cuda()
    output = model(data)
    loss = crit(output, target)

    train_loss += loss.item()
    _, pre = output.max(1)
    correct += sum(pre == target)
    total += target.size(0)
print(f"train_loss:{train_loss} | accuracy:{correct/total*100:.4f}%")