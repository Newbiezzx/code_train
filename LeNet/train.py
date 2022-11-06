import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import argparse
import matplotlib.pyplot as plt

from model import LeNet

# 设置超参数
parser =argparse.ArgumentParser(description="LeNet traning parameters")

parser.add_argument("--batch_size", '-b', default=64, type=int, help="batch sizes")
parser.add_argument("--lr", default=0.1, type=float, help="learning rate")
parser.add_argument("--epoch_max", default=10, type=int, help="epoch_max")
parser.add_argument("--plot", default=True, type=bool, help="whether plot")
args = parser.parse_args()

datatrain = MNIST("data", train=True, transform=(transforms.Compose(
        [transforms.Resize((32, 32)),
        transforms.ToTensor()]
    )), download=True)


# 加载数据集
train_loader = DataLoader(datatrain, args.batch_size, shuffle=True)
# 初始化模型
model = LeNet().cuda()
model.train()           # 切换为训练模型
optim = torch.optim.Adam(model.parameters(), lr=args.lr)        # 定义优化器
crit = nn.CrossEntropyLoss()     # 损失函数
epoch_max = args.epoch_max       # 训练轮次
Loss_lst = []                    # 构建损失值列表
Acc_lst = []                     # 构建准确率列表


for epoch in range(epoch_max):
    train_loss = 0
    correct = 0
    total = 0
    for _, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        optim.zero_grad()
        output = model(data)
        loss = crit(output, target)
        loss.backward()
        optim.step()

        train_loss += loss.item()
        _, pre = output.max(1)
        correct += sum(pre == target)
        total += target.size(0)
    print(f"epoch:{epoch} | train_loss:{train_loss} | accuracy:{correct/total*100:.4f}%")
    Loss_lst.append(train_loss)
    Acc_lst.append((correct/total*100).item())

# 保存模型参数
# torch.save(model, "./LeNet/LeNet_model.pth")    # 保存整个模型
torch.save(model.state_dict(), f"LeNet/LeNet_para_epoch{epoch}.pth")    # 只保存模型参数 

if args.plot:
    plt.rcParams['font.sans-serif']=['SimHei']#用来正常显示中文标签
    plt.rcParams['axes.unicode_minus']=False#用来正常显示负号
    plt.subplot(1, 2, 1)
    plt.title("LeNet train loss")
    plt.plot(Loss_lst, label="loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.title("LeNet train accuracy")
    plt.plot(Acc_lst, label="accuracy")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.savefig("train.png")
    plt.show()