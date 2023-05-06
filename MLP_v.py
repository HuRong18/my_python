import torch
import torch.nn as nn
from gpu_project_hr.python_project.pythonProject_values import const


class MLP(nn.Module):

    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()

        # self.fc6 = nn.Linear(in_features=3 * const.dimension * const.dimension,
        #                      out_features=128 * const.dimension * const.dimension)
        # self.fc7 = nn.Linear(in_features=128 * const.dimension * const.dimension,
        #                      out_features=128* const.dimension * const.dimension)
        # self.fc8 = nn.Linear(in_features=128 * const.dimension * const.dimension,
        #                      out_features=128 * const.dimension * const.dimension)
        # self.fc9 = nn.Linear(in_features=128 * const.dimension * const.dimension,
        #                      out_features=1 * const.dimension * const.dimension_mm)

        # self.fc6 = nn.Linear(in_features=3 * const.dimension * const.dimension,
        #                      out_features=1000)
        # self.fc7 = nn.Linear(in_features=1000,
        #                      out_features=2000)
        # self.fc8 = nn.Linear(in_features=2000,
        #                      out_features=1000)
        # self.fc9 = nn.Linear(in_features=1000,
        #                      out_features=1 * const.dimension * const.dimension_mm)

        self.fc6 = nn.Linear(in_features=const.input_k * const.dimension * const.dimension,
                             out_features=100)
        self.fc7 = nn.Linear(in_features=100,
                             out_features=100)
        self.fc8 = nn.Linear(in_features=100,
                             out_features=1 * const.dimension * const.dimension_mm)

    def activation_function(self, x: torch.Tensor):
        return x * torch.sigmoid(x)

    def forward_(self, x):
        x = self.fc8(self.activation_function(self.fc7(self.activation_function(self.fc6(self.flatten(x))))))
        x = x.view(-1, 1, const.dimension, const.dimension_mm)
        return x
