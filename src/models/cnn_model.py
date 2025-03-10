import torch
from torch import nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self, classes, dropout_prob=.5):
        super(CNNModel, self).__init__()
        # kernel
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        # dropout
        self.dropout = nn.Dropout(dropout_prob)
        # affine
        self.fc1 = nn.Linear(32 * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, classes)
        # others
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.maxpool(self.relu(self.conv1(x))) # 1, 28, 28 -> 16, 28, 28 -> 16, 14, 14
        x = self.maxpool(self.relu(self.conv2(x))) # 16, 14, 14 -> 32, 14, 14 -> 32, 7, 7 
        x = self.dropout(x)
        x = x.view(x.size()[0], -1) # [batch, channel, height, width] -> [batch, channel * height * width]
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x