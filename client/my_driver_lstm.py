from pytocl.driver import Driver
from pytocl.car import State, Command
from ESN import ESN
import numpy as np
import pickle
import math

import torch
import torch.nn as nn
from torch.autograd import Variable


def load_obj(name):
    with open('parameters/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=False
        )
        self.out = nn.Linear(hidden_size, output_size)
        self.hidden = self.init_hidden()

    def init_hidden(self, x=None):
        if x is None:
            return (Variable(torch.zeros(self.num_layers, 1, self.hidden_size)),
                    Variable(torch.zeros(self.num_layers, 1, self.hidden_size)))
        else:
            return Variable(x[0].data), Variable(x[1].data)

    def forward(self, x):
        lstm_out, hidden_out = self.lstm(x, self.hidden)
        output = self.out(lstm_out.view(len(x), -1))
        self.hidden = self.init_hidden(hidden_out)
        return output


# noinspection PyPep8Naming,PyMethodMayBeStatic
class MyDriver(Driver):

    def __init__(self, logdata=True):
        super().__init__(logdata)

        self.neural_net = LSTM(28, 56, 3, 3)
        self.neural_net.load_state_dict(torch.load('parameters/rnn_params.pt'))

        self.params_dict = load_obj('norm_parameters')

    def normalize(self, x, mmin, mmax):
        return np.clip((x - mmin) / (mmax - mmin), 0, 1)

    def invert_normalize(self, x, mmin, mmax):
        x = np.clip(x, -1, 1)
        return x * (mmax - mmin) + mmin

    def scale_array(self, x, mmin=0.0, mmax=1.0, new_min=-1.0, new_max=1.0):
        x = self.normalize(x, mmin, mmax)
        return np.clip(x * (new_max - new_min) + new_min, new_min, new_max)

    def drive(self, carstate: State) -> Command:
        X = np.array([
            self.scale_array(carstate.speed_x, self.params_dict['minSpeedX'], self.params_dict['maxSpeedX']),
            self.scale_array(carstate.speed_y, self.params_dict['minSpeedY'], self.params_dict['maxSpeedY']),
            self.scale_array(carstate.angle, -math.pi, math.pi),
            self.scale_array(carstate.gear, -1, 6),
            self.scale_array(carstate.rpm, 0, self.params_dict['maxRPM'])
        ])
        wheelSpin = [self.scale_array(i, 0, self.params_dict['maxWheelSpin'])
                     for i in list(carstate.wheel_velocities)]

        distFromEdge = [self.scale_array(i, 0, 200)
                        for i in list(carstate.distances_from_edge)]

        # X = np.array([
        #     carstate.speed_x,
        #     carstate.speed_y,
        #     carstate.angle,
        #     carstate.gear,
        #     carstate.rpm
        # ])
        # wheelSpin = [i for i in list(carstate.wheel_velocities)]
        # distFromEdge = [i for i in list(carstate.distances_from_edge)]
        #
        # X = np.concatenate((X, wheelSpin, distFromEdge))

        X = np.concatenate((X, wheelSpin, distFromEdge))

        X = torch.from_numpy(X).float()
        params = Variable(X.view(-1, 1, 28))
        output = self.neural_net(params)

        results = output.resize(3).data.numpy()
        steer = results[0]
        acc = results[1]
        brake = results[2]

        command = Command()

        acc = self.normalize(acc, -1, 1)
        brake = self.normalize(brake, -1, 1)

        if acc > 0:
            command.brake = 0
            command.accelerator = acc

            if carstate.rpm > 8000:
                command.gear = carstate.gear + 1
        else:
            command.brake = brake
            command.accelerator = 0

            if carstate.rpm < 2500:
                command.gear = carstate.gear - 1

        if not command.gear:
            command.gear = carstate.gear or 1

        print('{} - {}'.format(steer, self.scale_array(steer, 0, 1)))
        command.steering = self.scale_array(steer, 0, 1)

        # results = output.resize(2).data.numpy()
        # # gear = results[0]
        # steer = results[0]
        # accel_brake = results[1]
        #
        # command = Command()
        #
        # accel_brake = min(max(0, accel_brake), 1)
        # if accel_brake >= 0.5:
        #     command.brake = 0
        #     command.accelerator = self.scale_array(accel_brake, 0.5, 1, 0, 1)
        #
        #     if carstate.rpm > 8000:
        #         command.gear = carstate.gear + 1
        # else:
        #     command.brake = self.scale_array(accel_brake, 0.0, 0.5, 0, 1)
        #     command.accelerator = 0
        #
        #     if carstate.rpm < 2500 and carstate.gear != 1:
        #         command.gear = carstate.gear - 1

        return command
