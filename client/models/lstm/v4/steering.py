
# coding: utf-8

# In[1]:


import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

CUDA = torch.cuda.is_available()
if CUDA:
    DTYPE = torch.cuda.FloatTensor
else:
    DTYPE = torch.FloatTensor

# In[2]:


class Steering(nn.Module):
    def __init__(self, input_size, lstm_hidden_size, hidden_layer_size, dropout_prob, num_layers, output_size, batch_size):
        super(Steering, self).__init__()
        self.input_size = input_size
        self.lstm_hidden_size = lstm_hidden_size
        self.hidden_layer_size = hidden_layer_size
        self.dropout_prob = dropout_prob
        
        self.num_layers = num_layers
        self.output_size = output_size
        self.batch_size = batch_size
        
        self.hidden_state = self.init_hidden()

        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.lstm_hidden_size, num_layers=self.num_layers)
        
        self.hidden = nn.Linear(self.lstm_hidden_size, self.hidden_layer_size)
        
        self.tanh = nn.Tanh()
        
        self.dropout = nn.Dropout(p=dropout_prob)
        
        self.out = nn.Linear(self.hidden_layer_size, self.output_size)

    def init_hidden(self):
        return (autograd.Variable(torch.zeros(self.num_layers, self.batch_size, self.lstm_hidden_size)).type(DTYPE),
                    autograd.Variable(torch.zeros(self.num_layers, self.batch_size, self.lstm_hidden_size)).type(DTYPE))
    
    def forward(self, x):
        lstm_out, self.hidden_state = self.lstm(x.view(len(x), 1, -1), self.hidden_state)
        
        hidden_out = self.tanh(self.hidden(lstm_out))

        drop_out = self.dropout(hidden_out)
        
        out = self.out(drop_out)
        
        return out
    
class HYPERPARAMS(object):
    TARGET_SIZE = 1
    INPUT_SIZE = 22
    LSTM_HIDDEN_SIZE = 200
    HIDDEN_LAYER_SIZE = 100
    DROPOUT_PROB = 0.2
    NUM_LAYERS = 2
    
    
    BATCH_SIZE = 1
    LEARNING_RATE = 1e-3
    