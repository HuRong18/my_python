import torch
import torch.nn as nn
import torch.nn.functional as F
from gpu_project_hr.python_project.pythonProject_values import const
from gpu_project_hr.python_project.pythonProject_values import MLP_re
from gpu_project_hr.python_project.pythonProject_values import util
from gpu_project_hr.python_project.pythonProject_values import MLP_v0
from gpu_project_hr.python_project.pythonProject_values import MLP_v


class DNN(nn.Module):

    def __init__(self, alpha_list, device):
        super().__init__()
        self.mlp_0 = MLP_re.MLP()
        self.mlp_1 = MLP_re.MLP()
        self.mlp_2 = MLP_re.MLP()
        # self.mlp_9 = MLP_re.MLP()
        # self.mlp_10 = MLP_re.MLP()
        # self.mlp_11 = MLP_re.MLP()
        # self.mlp_12 = MLP_re.MLP()
        # self.mlp_13 = MLP_re.MLP()
        # self.mlp_14 = MLP_re.MLP()
        # self.mlp_14 = MLP_re.MLP()
        # self.mlp_15 = MLP_re.MLP()
        # self.mlp_16 = MLP_re.MLP()
        self.mlp_3 = MLP_v0.MLP()
        self.mlp_4 = MLP_v.MLP()
        self.mlp_5 = MLP_v.MLP()
        # self.mlp_6 = MLP_v.MLP()
        # self.mlp_7 = MLP_v.MLP()
        # self.mlp_8 = MLP_v.MLP()

        self.optimizer = torch.optim.Adam(self.parameters(), 1e-4)
        self.loss_function = F.mse_loss
        self.alpha_list = alpha_list
        self.device = device
        self.to(device)

    def forward(self, x):
        length = len(x)
        u_list1 = [self.mlp_0.forward(x), self.mlp_1.forward(x), self.mlp_2.forward(x)]
        u_list2 = [self.mlp_4.forward_(x), self.mlp_5.forward_(x)]
        u0 = self.mlp_3.forward_V0(x)
        res = []
        for alpha in self.alpha_list:
            y = torch.zeros(length, 1, const.dimension, const.dimension).to(self.device)
            code = torch.zeros(length, 1, const.dimension, const.dimension_s).to(self.device)
            tmp = torch.zeros(length, 1, const.dimension, const.dimension_s)
            for index, u in enumerate(u_list1):
                y.add_(u * alpha ** index)
            y = y.reshape(-1, 1, const.dimension_mm, 1)
            for index_, u_ in enumerate(u_list2):
                z = y ** (index_ + 1)
                code = torch.matmul(u_, z) \
                    .reshape(length,1,const.dimension,const.dimension_s)
                tmp = code + tmp
                # for i in range(32):
                #     for j in range(1):
                #         code.add_(u_[i][j] * z[i][j])
            tmp.add_(u0)
            res.append(tmp)
        # res = torch.cat(res, 1).reshape(32, 3, 100, 1)
        # code=[]
        # for i in range(32):
        #     tmp = torch.zeros(length, 1, const.dimension, 1).to(self.device)
        #     for j in range(3):
        #         for index, u in enumerate(u_list2):
        #             u=u.reshape(1, -1)
        #             res=res.reshape(-1, 1)
        #             tmp.add_(u * res ** index)
        #     code.append(tmp)
        res = torch.cat(res, 1)
        # res=res.reshape(32, 3, const.dimension, const.dimension)
        return res

    def train_by_data(self, input: torch.Tensor, features: torch.Tensor):
        self.train()
        self.optimizer.zero_grad()
        out = self.forward(input)
        loss = self.loss_function(input=out, target=features, reduction='sum')
        loss.backward()
        self.optimizer.step()
        return loss
