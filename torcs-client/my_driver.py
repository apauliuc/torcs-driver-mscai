from pytocl.driver import Driver
from pytocl.car import State, Command

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        
        self.hidden_size = 200
        self.num_layers = 6
        
        self.lstm = nn.LSTM(
            input_size=22,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(self.hidden_size, 3)
    
    def forward(self, x):
        h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
        
        out, _ = self.lstm(x, (h0, c0))
        
        out = self.fc(out[:, -1, :])
        return out


class MyDriver(Driver):

    def __init__(self, logdata=True):
        super().__init__(logdata)

        self.neural_net = RNN()
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
        
        if acc > 1:
            command.accelerator = 1
        elif acc < 0:
            command.accelerator = 0
        else:
            command.accelerator = acc

        if brake > 1:
            command.brake = 1
        elif brake < 0:
            command.brake = 0
        else:
            command.brake = brake

        if steer > 1:
            command.steering = 1
        elif steer < -1:
            command.steering = -1
        else:
            command.steering = steer

        if carstate.rpm > 8000:
            command.gear = carstate.gear + 1
        
        if carstate.rpm < 2500:
            command.gear = carstate.gear - 1
        
        if not command.gear:
            command.gear = carstate.gear or 1
        
        return command
