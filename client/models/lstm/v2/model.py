
# coding: utf-8

# In[1]:


import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler


# In[2]:


class LSTMDriver(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size):
        super(LSTMDriver, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.batch_size = batch_size
        
        super(LSTMDriver, self).__init__()

        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers)
        
        self.hidden = self.init_hidden()
        
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def init_hidden(self):
        return (autograd.Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_size)),
                    autograd.Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_size)))
    
    def forward(self, x):
        lstm_out, self.hidden = self.lstm(x.view(len(x), 1, -1), self.hidden)
        linear_out = self.linear(lstm_out)
        
        return linear_out

