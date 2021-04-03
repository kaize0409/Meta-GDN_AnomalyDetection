import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def deviation_loss(y_true, y_prediction):
    '''
    z-score based deviation loss
    :param y_true: true anomaly labels
    :param y_prediction: predicted anomaly label
    :return: loss in training
    '''
    confidence_margin = 5.0
    ref = torch.tensor(np.random.normal(loc=0.0, scale=1.0, size=5000), dtype=torch.float32)
    dev = (y_prediction - torch.mean(ref)) / torch.std(ref)
    inlier_loss = torch.abs(dev)
    outlier_loss = confidence_margin - dev
    outlier_loss[outlier_loss < 0.] = 0.
    return torch.mean((1 - y_true) * inlier_loss + y_true * outlier_loss)
    # pass

class FCNet(nn.Module):
    def __init__(self, in_feature, out_feature):
        super(FCNet, self).__init__()
        self.fc1 = nn.Linear(in_feature, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, out_feature)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(256)
        self.batchnorm2 = nn.BatchNorm1d(64)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.batchnorm1(x)
        x = self.relu(self.fc2(x))
        x = self.batchnorm2(x)
        x = self.dropout(x)

        return self.fc3(x)

class SGC(nn.Module):

    def __init__(self, in_feature, out_feature):
        super(SGC, self).__init__()
        # mid_feature = 1 / (in_feature + out_feature)
        self.fc1 = nn.Linear(in_feature, 512)
        # self.fc2 = nn.Linear(512, 256)
        self.dropout = nn.Dropout(p=0.1)
        # self.W = nn.Linear(in_feature, out_feature)
        # self.W = nn.Linear(in_feature, mid_feature)
        self.out = nn.Linear(512, out_feature)
    def forward(self, x):
        x = self.fc1(x)
        # x = self.dropout(x)
        # x = F.relu(self.fc1(x))
        # x = self.dropout(x)
        # x = F.relu(self.fc2(x))
        x = self.out(x)
        return x
