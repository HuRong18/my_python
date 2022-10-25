import torch
import torch.nn as nn
import const


class MLP(nn.Module):

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(in_features=3 * const.dimension * const.dimension,
                             out_features=16 * const.dimension * const.dimension)
        self.fc2 = nn.Linear(in_features=16 * const.dimension * const.dimension,
                             out_features=8 * const.dimension * const.dimension)
        self.fc3 = nn.Linear(in_features=8 * const.dimension * const.dimension,
                             out_features=1 * const.dimension * const.dimension)

        self.fc7 = nn.Linear(in_features=8 * const.dimension * const.dimension,
                             out_features=1 * const.dimension_mm * 1)

        self.fc4 = nn.Linear(in_features=3 * const.dimension * const.dimension,
                             out_features=16 * const.dimension * const.dimension)
        self.fc5 = nn.Linear(in_features=16 * const.dimension * const.dimension,
                             out_features=2 * const.dimension_mm * const.dimension_mm)
        self.fc6 = nn.Linear(in_features=2 * const.dimension_mm * const.dimension_mm,
                             out_features=1 * const.dimension_mm * const.dimension_mm)

    def forward(self, x):
        x = self.fc3(torch.tanh(self.fc2(torch.tanh(self.fc1(self.flatten(x))))))
        x = x.view(-1, 1, const.dimension, const.dimension)
        return x

    def forward_V0(self, x):
        x = self.fc7(torch.tanh(self.fc2(torch.tanh(self.fc1(self.flatten(x))))))
        x = x.view(-1, 1, const.dimension_mm, 1)
        return x

    def forward_(self, x):
        x = self.fc6(torch.tanh(self.fc5(torch.tanh(self.fc4(self.flatten(x))))))
        x = x.view(-1, 1, const.dimension_mm, const.dimension_mm)
        return x
