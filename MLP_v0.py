import torch
import torch.nn as nn
from gpu_project_hr.python_project.pythonProject_values import const


class MLP(nn.Module):

    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()

        # self.fc1 = nn.Linear(in_features=3 * const.dimension * const.dimension,
        #                      out_features=128 * const.dimension * const.dimension)
        # self.fc2 = nn.Linear(in_features=128 * const.dimension * const.dimension,
        #                      out_features=256 * const.dimension * const.dimension)
        # self.fc3 = nn.Linear(in_features=256 * const.dimension * const.dimension,
        #                      out_features=128 * const.dimension * const.dimension)
        # self.fc4 = nn.Linear(in_features=128 * const.dimension * const.dimension,
        #                      out_features=1 * const.dimension * 1)

        # self.fc1 = nn.Linear(in_features=3 * const.dimension * const.dimension,
        #                      out_features=1000)
        # self.fc2 = nn.Linear(in_features=1000,
        #                      out_features=2000)
        # self.fc3 = nn.Linear(in_features=2000,
        #                      out_features=1000)
        # self.fc4 = nn.Linear(in_features=1000,
        #                      out_features=1 * const.dimension * 1)

        self.fc1 = nn.Linear(in_features=const.input_k * const.dimension * const.dimension,
                             out_features=100)
        self.fc2 = nn.Linear(in_features=100,
                             out_features=100)
        self.fc3 = nn.Linear(in_features=100,
                             out_features=1 * const.dimension* const.dimension_s)


    def activation_function(self, x: torch.Tensor):
        return x * torch.sigmoid(x)

    def forward_V0(self, x):
        # x = self.fc4(self.activation_function(self.fc3(self.activation_function
        #                                                (self.fc2(
        #                                                    self.activation_function(self.fc1(self.flatten(x))))))))
        x = self.fc3(self.activation_function
                                                       (self.fc2(
                                                           self.activation_function(self.fc1(self.flatten(x))))))
        x = x.view(-1, 1, const.dimension, const.dimension_s)
        return x

