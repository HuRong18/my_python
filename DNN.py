import torch
import torch.nn as nn
import torch.nn.functional as F
import MLP
import const

class DNN(nn.Module):

    def __init__(self, alpha_list, device):
        super().__init__()  # 调用父类
        torch.set_default_dtype(torch.float64)
        self.mlp_0 = MLP.MLP()
        self.mlp_1 = MLP.MLP()
        self.mlp_2 = MLP.MLP()
        self.mlp_3 = MLP.MLP()

        self.optimizer = torch.optim.Adam(self.parameters(), 1e-4)
        self.loss_function = F.l1_loss
        self.alpha_list = alpha_list

        self.device = device
        self.to(device)

    def forward(self, x):
        # 过神经网络
        torch.set_default_dtype(torch.float64)
        length = len(x)
        u_list = [self.mlp_0.forward(x), self.mlp_1.forward(x), self.mlp_2.forward(x), self.mlp_3.forward(x)]
        # 计算多项式
        res = []
        for alpha in self.alpha_list:
            y = torch.zeros(length, 1, const.dimension, 1).to(self.device)
            for index, u in enumerate(u_list):
                y.add_(u * alpha ** index)
            res.append(y)
        return torch.cat(res, 1)

    def train_by_data(self, input: torch.Tensor, feature: torch.Tensor):
        self.train()
        torch.set_default_dtype(torch.float64)
        self.optimizer.zero_grad()
        out = self.forward(input)
        loss = self.loss_function(input=out, target=feature, reduction='sum')
        loss.backward()
        self.optimizer.step()
        return loss
