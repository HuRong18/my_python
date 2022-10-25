import torch
import torch.nn as nn
import torch.nn.functional as F

import const


class MLP(nn.Module):

    def __init__(self):
        super().__init__()  # 调用父类
        self.flatten = nn.Flatten()
        torch.set_default_dtype(torch.float64)

        self.fc1 = nn.Linear(in_features=3 * const.dimension * const.dimension, out_features=4 * const.dimension * const.dimension)
        self.fc2 = nn.Linear(in_features=4 * const.dimension * const.dimension, out_features=8 * const.dimension * const.dimension)
        self.fc3 = nn.Linear(in_features=8 * const.dimension * const.dimension, out_features=64 * const.dimension * 1)
        self.fc4 = nn.Linear(in_features=64 * const.dimension * 1, out_features=32 * const.dimension * 1)
        self.fc5 = nn.Linear(in_features=32 * const.dimension * 1, out_features=16 * const.dimension * 1)
        self.fc6 = nn.Linear(in_features=16 * const.dimension * 1, out_features=4 * const.dimension * 1)
        self.fc7 = nn.Linear(in_features=4 * const.dimension * 1, out_features=1 * const.dimension * 1)
        # for m in self.modules():

    #     if isinstance(m, nn.Linear):
    #         nn.init.normal_(m.weight.data, 0, 0.0001)
    #         m.bias.data.zero_()

    def forward(self, x):
        # 过神经网络
        torch.set_default_dtype(torch.float64)
        x = torch.sigmoid(self.fc4(torch.sigmoid(self.fc3(torch.sigmoid(self.fc2(torch.sigmoid
                                                                                 (self.fc1(self.flatten(x)))))))))
        x = self.fc7(torch.sigmoid(self.fc6(torch.sigmoid(self.fc5(x)))))
        # x = self.fc6(F.relu(x))
        x = x.view(-1, 1, const.dimension, 1)
        return x

    # def initialize_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Linear):
    #             nn.init.kaiming_normal_(m.weight.data, 0, mode='fan_in', nonlinearity='relu')
    #             m.bias.data.zero_()
