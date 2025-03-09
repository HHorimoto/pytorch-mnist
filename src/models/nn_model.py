import torch
from torch import nn

class NNModel(nn.Module):
    def __init__(self, classes, dropout_prob=.5):
        super(NNModel, self).__init__()
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)

        self.linear1 = nn.Linear(28*28, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, classes)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.linear1(x))
        x = self.dropout(x)

        x = self.relu(self.linear2(x))
        x = self.dropout(x)

        logits = self.linear3(x)
        return logits