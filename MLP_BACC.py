import torch
import torch.nn as nn
import const


class MLP(nn.Module):

    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()

        # self.fc1 = nn.Linear(in_features=3 * const.dimension * const.dimension,
        #                      out_features=128* const.dimension * const.dimension)
        # self.fc2 = nn.Linear(in_features=128 * const.dimension * const.dimension,
        #                      out_features=128 * const.dimension * const.dimension)
        # self.fc3 = nn.Linear(in_features=128 * const.dimension * const.dimension,
        #                      out_features=128 * const.dimension * const.dimension)
        # self.fc4 = nn.Linear(in_features=128 * const.dimension * const.dimension,
        #                      out_features=1 * const.dimension * const.dimension)

        # self.fc1 = nn.Linear(in_features=3 * const.dimension * const.dimension,
        #                      out_features=1000)
        # self.fc2 = nn.Linear(in_features=1000,
        #                      out_features=2000)
        # self.fc3 = nn.Linear(in_features=2000,
        #                      out_features=1000)
        # self.fc4 = nn.Linear(in_features=1000,
        #                      out_features=1 * const.dimension * const.dimension)

        self.fc1 = nn.Linear(in_features=const.input_k_BACC * const.dimension * const.dimension,
                             out_features=100)
        self.fc2 = nn.Linear(in_features=100,
                             out_features=100)
        self.fc3 = nn.Linear(in_features=100,
                             out_features=1 * const.dimension * const.dimension)

    def activation_function(self, x: torch.Tensor):
        return x * torch.sigmoid(x)

    def forward(self, x):
        # x = self.fc4(self.activation_function(self.fc3(self.activation_function
        #                                                (self.fc2(self.activation_function(self.fc1(self.flatten(x))))))))
        x = self.fc3(self.activation_function
            (self.fc2(
            self.activation_function(self.fc1(self.flatten(x))))))
        x = x.view(-1, 1, const.dimension, const.dimension)
        return x

