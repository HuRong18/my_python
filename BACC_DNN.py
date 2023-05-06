import torch
import torch.nn as nn
import torch.nn.functional as F
import const
import MLP_BACC
import util
import MLP_v0
import MLP_v


class DNN(nn.Module):

    def __init__(self, alpha_list, device):
        super().__init__()
        self.mlp_0 = MLP_BACC.MLP()
        self.mlp_1 = MLP_BACC.MLP()
        self.mlp_2 = MLP_BACC.MLP()
        # self.mlp_6 = MLP_BACC.MLP()
        # self.mlp_7 = MLP_BACC.MLP()
        self.mlp_3 = MLP_v0.MLP()
        self.mlp_4 = MLP_v.MLP()
        self.mlp_5 = MLP_v.MLP()

        self.optimizer = torch.optim.Adam(self.parameters(), 1e-4)
        self.loss_function = F.mse_loss
        self.alpha_list = alpha_list
        self.device = device
        self.to(device)

    def forward(self, x):
        length = len(x)
        u_list1 = [self.mlp_0.forward(x), self.mlp_1.forward(x), self.mlp_2.forward(x)]
        res = []
        for alpha in self.alpha_list:
            y = torch.zeros(length, 1, const.dimension, const.dimension).to(self.device)
            for index, u in enumerate(u_list1):
                y.add_(u * alpha ** index)
            res.append(y)
        res = torch.cat(res, 1)
        # res=res.reshape(32, 3, const.dimension, const.dimension)
        return res

    def forward_compute(self, x, res):
        u_list2 = [self.mlp_4.forward_(x), self.mlp_5.forward_(x)]
        u0 = self.mlp_3.forward_V0(x)
        res = res.reshape(-1, len(self.alpha_list), const.dimension_mm, 1)
        tmp = []
        for i in range(len(self.alpha_list)):
            code = torch.zeros(len(x), 1, const.dimension, 1).to(self.device)
            for index_, u_ in enumerate(u_list2):
                z = res[:, i].unsqueeze(1) ** (index_ + 1)
                code = torch.matmul(u_, z)
            code.add_(u0)
            tmp.append(code)
        tmp = torch.cat(tmp, 1)
        return tmp

    def train_by_data_compute(self, input: torch.Tensor, output: torch.tensor, features: torch.Tensor):
        self.train()
        self.optimizer.zero_grad()
        out = self.forward_compute(input, output)
        loss_compute = self.loss_function(input=out, target=features, reduction='sum')
        loss_compute.backward()
        self.optimizer.step()
        return loss_compute

    def train_by_data(self, input: torch.Tensor, features: torch.Tensor):
        self.train()
        self.optimizer.zero_grad()
        out = self.forward(input)
        loss = self.loss_function(input=out, target=features, reduction='sum')
        loss.backward()
        self.optimizer.step()
        return loss
