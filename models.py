import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SGC(nn.Module):

    def __init__(self, in_feature, out_feature):
        super(SGC, self).__init__()
        self.fc1 = nn.Linear(in_feature, 512)
        self.out = nn.Linear(512, out_feature)

    def forward(self, x):
        x = self.fc1(x)
        x = self.out(x)
        return x
