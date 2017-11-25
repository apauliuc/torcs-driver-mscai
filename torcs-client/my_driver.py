from pytocl.driver import Driver
from pytocl.car import State, Command

import numpy as np
import pickle
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

        self.neural_net = RNN(28, 28, 3, 3)
        self.neural_net.load_state_dict(torch.load('rnn_params.pt'))
        with open('norm_parameters.pickle', 'rb') as handle:
            self.params_dict = pickle.load(handle)

    def normalize(x, min, max):
        return (x - min)/(max-min)

    def invert_normalize(x, min, max):
        return x * (max - min) + min

    def drive(self, carstate: State) -> Command:
        wheelSpin = [self.normalize(i, 0, self.params_dict['maxWheelSpin'] for i in list(carstate.wheel_velocities))]
        distFromEdge = [self.normalize(i, 0, 200) for i in list(carstate.distances_from_edge)]
        X = np.array([
            self.normalize(carstate.speed_x, self.params_dict['minSpeedX'], self.params_dict['maxSpeedX']),
            self.normalize(carstate.speed_y, self.params_dict['minSpeedY'], self.params_dict['maxSpeedY']),
            self.normalize(carstate.angle, -180, 180),
            self.normalize(carstate.gear, -1, 6),
            self.normalize(carstate.rpm, 0, self.params_dict['maxRPM'])
        ] + wheelSpin + distFromEdge

        X = torch.from_numpy(X).float()
        params = Variable(X.view(-1, 1, 22))
        output = self.neural_net(params)

        results = output.resize(3).data.numpy()        
        gear = results[0]
        steer = results[1]
        accel_brake = results[2]

        command = Command()
        
        accel_brake = max(min(0, accel_brake), 1)
        if accel_brake >= 0.5:
            command.brake = 0
            command.accelerator = self.normalize(accel_brake, 0.5, 1)
        else:
            command.brake = self.normalize(accel_brake, 0, 0.5)
            command.accelerator = 0

        gear = max(min(0, gear), 1)
        gear = self.invert_normalize(gear, -1, 6)
        command.gear = int(round(gear))

        steer = max(min(0, steer), 1)
        steer = self.invert_normalize(steer, -1, 1)
        
        return command
