import torch
from torch import nn
import torch.optim as optim
class CountsNet(nn.Module):
    def __init__(self):
        super(CountsNet, self).__init__()
        self.linear1 = nn.Linear(256, 512)
        self.batchnorm1 = nn.BatchNorm1d(512)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.25)
        self.linear2 = nn.Linear(512,1024)
        self.batchnorm2 = nn.BatchNorm1d(1024)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.25)
        self.linear3 = nn.Linear(1024,1)
        self.sig = nn.Sigmoid()
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.linear3(x)
        x = self.sig(x)
        return x