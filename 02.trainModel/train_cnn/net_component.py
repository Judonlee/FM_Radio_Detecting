# -*- coding: utf-8 -*-

"""
@Author : zhudong
@Email  : ynzhudong@163.com
@Time   : 2019/3/29 下午12:21
@File   : net_component.py
desc:
"""

import torch
import torch.nn as nn


# Convolutional neural network
class FM_DeepModel(nn.Module):
    def __init__(self, num_classes=2):
        super(FM_DeepModel, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(5, 11), stride=(2, 5)),
            nn.BatchNorm2d(16),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(5, 11), stride=(2, 5)),
            nn.BatchNorm2d(32),
            nn.ReLU())
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.fc1 = nn.Linear(256, 256)
        self.preluip = nn.PReLU()
        self.fc2 = nn.Linear(256, num_classes)

    def l2_norm(self, input):
        input_size = input.size()
        buffer = torch.pow(input, 2)

        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)

        _output = torch.div(input, norm.view(-1, 1).expand_as(input))
        output = _output.view(input_size)
        return output

    def forward(self, x):
        x = self.layer1(x)

        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = x.reshape(x.size(0), -1)

        x = self.preluip(self.fc1(x))

        x = self.l2_norm(x)

        alpha = 10
        x = x * alpha

        out = self.fc2(x)

        return x, out   # return x for visualization