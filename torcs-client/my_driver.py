from pytocl.driver import Driver
from pytocl.car import State, Command

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        
        super(RNN, self).__init__()        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=False
        )
        self.out = nn.Linear(hidden_size, output_size)        
        self.hidden = self.init_hidden()
        
    def init_hidden(self, x=None):
        if x == None:
            return (Variable(torch.zeros(self.num_layers, 1, self.hidden_size)),
                    Variable(torch.zeros(self.num_layers, 1, self.hidden_size)))
        else:
            return (Variable(x[0].data),Variable(x[1].data))
        
    def forward(self, x):
        lstm_out, self.hidden_out = self.lstm(x, self.hidden)
        output = self.out(lstm_out.view(len(x), -1))
        self.hidden = self.init_hidden(self.hidden_out)
        return output

class MyDriver(Driver):

    def __init__(self, logdata=True):
        super().__init__(logdata)

        self.neural_net = RNN(22, 160, 3, 3)
        self.neural_net.load_state_dict(torch.load('rnn_params.pt'))

    def drive(self, carstate: State) -> Command:
        X = np.array([
            carstate.speed_x,
            carstate.distance_from_center,
            carstate.angle        
        ] + list(carstate.distances_from_edge))

        X = torch.from_numpy(X).float()
        params = Variable(X.view(-1, 1, 22))
        output = self.neural_net(params)

        results = output.resize(3).data.numpy()
        acc = results[0]
        brake = results[1]
        steer = results[2]

        command = Command()
        
        if acc > 0:
            if abs(carstate.distance_from_center) >= 1:
                acc = min(0.4, acc)
            
            command.accelerator = min(acc, 1)

            if carstate.rpm > 8000:
                command.gear = carstate.gear + 1
        else:
            command.accelerator = 0

        if carstate.rpm < 3000 and carstate.gear != 0:
            command.gear = carstate.gear - 1
        
        if not command.gear:
            command.gear = carstate.gear or 1

        command.brake = min(max(0, brake), 1)

        command.steering = min(max(-1, steer), 1)

        #print('acc: %.4f, brake: %.4f, steer: %.4f' %(command.accelerator, command.brake, command.steering))
        
        return command
