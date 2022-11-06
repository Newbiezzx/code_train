from sklearn.datasets import load_boston
import torch
import torch.nn as nn

class LinearModel(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_ch, 1))
        self.bias = nn.Parameter(torch.randn(1))

        
    def forward(self, x):
        return x.mm(self.weight) + self.bias


bosdon_x = torch.tensor(load_boston()["data"], requires_grad=True, dtype=torch.float32).cuda()
bosdon_y = torch.tensor(load_boston()["target"], dtype=torch.float32).cuda()
lm = LinearModel(13).cuda()
criterion = nn.MSELoss()
optim = torch.optim.Adam(lm.parameters())
for step in range(100000):
    pred = lm(bosdon_x)
    loss = criterion(pred, bosdon_y)
    if step%100 == 0:
        print(f"第{step}次迭代的MSEloss为{loss}")
    optim.zero_grad()
    loss.backward()
    optim.step()