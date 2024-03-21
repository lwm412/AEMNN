import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

cuda = torch.cuda.is_available()
## Multilayer LSTM based classifier taking in 200 dimensional fixed time series inputs
class LSTMClassifier(nn.Module):

    def __init__(self, in_dim, hidden_dim, num_layers, dropout, bidirectional, num_classes, batch_size):
        super(LSTMClassifier, self).__init__()
        self.arch = 'lstm'
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_dir = 2 if bidirectional else 1
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
        )

        self.hidden2label = nn.Sequential(
            nn.Linear(hidden_dim * self.num_dir, hidden_dim),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(hidden_dim, num_classes),
        )

        # self.hidden = self.init_hidden()

    def init_hidden(self):
        if cuda:
            h0 = Variable(torch.zeros(self.num_layers * self.num_dir, self.batch_size, self.hidden_dim).cuda())
            c0 = Variable(torch.zeros(self.num_layers * self.num_dir, self.batch_size, self.hidden_dim).cuda())
        else:
            h0 = Variable(torch.zeros(self.num_layers * self.num_dir, self.batch_size, self.hidden_dim))
            c0 = Variable(torch.zeros(self.num_layers * self.num_dir, self.batch_size, self.hidden_dim))
        return (h0, c0)

    def forward(self, x):  # x is (batch_size, 28, 14), permute to (28, batch_size, 14)
        x = x.permute(1, 0, 2)
        # See: https://discuss.pytorch.org/t/solved-why-we-need-to-detach-variable-which-contains-hidden-representation/1426/2
        lstm_out, (h, c) = self.lstm(x, self.init_hidden())
        y = self.hidden2label(lstm_out[-1])
        return y
