import torch
import torch.nn as nn
import torch.nn.functional as F


class SEmodule(nn.Module):
    def __init__(self, num_channels, fc_size):
        super(SEmodule, self).__init__()
        self.global_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(num_channels, fc_size)
        self.fc2 = nn.Linear(fc_size, num_channels)

    def forward(self, x):
        se = self.global_pooling(x)
        se = F.relu(self.fc1(se))
        se = F.sigmoid(self.fc2(se))
        se = x * se

        return se
