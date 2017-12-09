from pytocl.driver import Driver
from pytocl.car import State, Command
from ESN import ESN
import numpy as np
import pickle
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

dtype = torch.FloatTensor


def load_obj(name):
    with open('parameters/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


class LSTMDriver(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size=1):
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
        return (Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_size),
                         requires_grad=False).type(dtype),
                Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_size),
                         requires_grad=False).type(dtype))

    def forward(self, x):
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        linear_out = self.linear(lstm_out)
        return linear_out


# noinspection PyPep8Naming,PyMethodMayBeStatic
class MyDriver(Driver):

    def __init__(self, logdata=True):
        super().__init__(logdata)

        # params = load_obj('rnn_params')

        params = {
            'input': 22,
            'hidden': 77,
            'layers': 2,
            'output': 3
        }

        self.neural_net = LSTMDriver(
            params['input'],
            params['hidden'],
            params['layers'],
            params['output']
        )
        self.neural_net.load_state_dict(torch.load('parameters/evolved'))
        self.neural_net.init_hidden()

        # self.means = load_obj('means')
        # self.stdevs = load_obj('stdevs')

    def normalize_to_0_1(self, x, mmin, mmax):
        return np.clip((x - mmin) / (mmax - mmin), 0, 1)

    def invert_normalize(self, x, mmin, mmax):
        x = np.clip(x, -1, 1)
        return x * (mmax - mmin) + mmin

    def scale_array(self, x, mmin=0.0, mmax=1.0, new_min=-1.0, new_max=1.0):
        x = self.normalize_to_0_1(x, mmin, mmax)
        return np.clip(x * (new_max - new_min) + new_min, new_min, new_max)

    def drive(self, carstate: State) -> Command:
        # X = np.array([
        #     self.scale_array(carstate.speed_x, -85, 360),
        #     self.scale_array(carstate.distance_from_center, -1, 1),
        #     self.scale_array(carstate.angle, -180, 180)
        # ])
        # distFromEdge = [self.scale_array(i, 0, 200)
        #                 for i in list(carstate.distances_from_edge)]
        #
        # X = np.concatenate((X, distFromEdge))

        # X = np.array([
        #     self.scale_array(carstate.speed_x, self.params_dict['minSpeedX'], self.params_dict['maxSpeedX']),
        #     self.scale_array(carstate.speed_y, self.params_dict['minSpeedY'], self.params_dict['maxSpeedY']),
        #     self.scale_array(carstate.angle, -math.pi, math.pi),
        #     self.scale_array(carstate.gear, -1, 6),
        #     self.scale_array(carstate.rpm, 0, self.params_dict['maxRPM'])
        # ])
        # wheelSpin = [self.scale_array(i, 0, self.params_dict['maxWheelSpin'])
        #              for i in list(carstate.wheel_velocities)]
        #
        # distFromEdge = [self.scale_array(i, 0, 200)
        #                 for i in list(carstate.distances_from_edge)]

        X = np.array([
            carstate.speed_x,
            carstate.distance_from_center,
            carstate.angle
        ])
        X = np.concatenate((X, list(carstate.distances_from_edge)))

        X = torch.from_numpy(X).float()
        params = Variable(X.view(1, -1))
        output = self.neural_net(params)

        # print(output.data)
        accelerator, brake, steer = output.data.numpy()[0][0]

        command = Command()

        command.accelerator = accelerator
        command.brake = brake

        # if accelerator - brake >= 0:
        #     command.brake = 0
        #     command.accelerator = 1.5 * (accelerator - brake)
        # else:
        #     command.brake = brake
        #     command.accelerator = 0

        if accelerator > 0:
            # command.brake = 0
            # command.accelerator = acc

            if carstate.rpm > 8000:
                command.gear = carstate.gear + 1
        else:
            # command.brake = brake
            # command.accelerator = 0

            if carstate.rpm < 2500:
                command.gear = carstate.gear - 1

        if not command.gear:
            command.gear = carstate.gear or 1

        # print('{} - {}'.format(steer, self.scale_array(steer, 0, 1)))
        command.steering = steer

        return command
