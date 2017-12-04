import glob
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size):
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.out = nn.Linear(hidden_size, output_size)
        self.hidden = self.init_hidden()

    def init_hidden(self, x=None):
        if x is None:
            return (Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_size)),
                    Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_size)))
        else:
            return Variable(x[0].data), Variable(x[1].data)

    def forward(self, x):
        lstm_out, hidden_out = self.lstm(x, self.hidden)
        output = self.out(lstm_out.view(len(x), -1))
        self.hidden = self.init_hidden(hidden_out)
        return output
        # return F.sigmoid(output)
