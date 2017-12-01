from pytocl.driver import Driver
from pytocl.car import State, Command
from finalESN import ESN

import numpy as np
import pickle
import pandas as pd

def load_obj(name):
    with open('parameters/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def safe_arctanh(x):
    if x.ndim == 2:
        for row in np.arange(x.shape[0]):
            x[row, :] = [i - 1e-15 if i == 1 else i + 1e-15 if i == -1 else i for i in x[row, :]]
    elif x.ndim == 1:
        x = [i - 1e-15 if i == 1 else i + 1e-15 if i == -1 else i for i in x]
    return np.arctanh(x)


# noinspection PyPep8Naming
class MyDriver(Driver):

    def __init__(self, logdata=True):
        super().__init__(logdata)

        self.esn = ESN(
            n_inputs=28,
            n_outputs=2,
            n_reservoir=200,
            spectral_radius=0.85,
            sparsity=0,
            noise=0.001,
            teacher_forcing=True,
            activation_out=np.tanh,
            inverse_activation_out=safe_arctanh,
            print_state=False
        )

        self.esn.random_state_ = load_obj('esn_random_state')
        last_data = load_obj('esn_last_data')
        weights = load_obj('esn_weights')

        self.esn.last_input = last_data['last_input']
        self.esn.last_state = last_data['last_state']
        self.esn.last_output = last_data['last_output']
        self.esn.W = weights['W']
        self.esn.W_in = weights['W_in']
        self.esn.W_feedback = weights['W_feedback']
        self.esn.W_out = weights['W_out']

        self.params_dict = load_obj('norm_parameters')

        self.first_step = False

    def normalize(self, x, mmin, mmax):
        norm = (x - mmin)/(mmax-mmin)
        return min(max(-1, norm), 1)

    def invert_normalize(self, x, mmin, mmax):
        x = min(max(-1, x), 1)
        return x * (mmax - mmin) + mmin

    def normalize_to_int(self, x, mmin, mmax, a=-1, b=1):
        return (b - a) * (x - mmin) / (mmax - mmin) + a

    def drive(self, carstate: State) -> Command:
        X = np.array([
            self.normalize(carstate.speed_x, self.params_dict['minSpeedX'], self.params_dict['maxSpeedX']),
            self.normalize(carstate.speed_y, self.params_dict['minSpeedY'], self.params_dict['maxSpeedY']),
            self.normalize(carstate.angle, -180, 180),
            self.normalize(carstate.gear, -1, 6),
            self.normalize(carstate.rpm, 0, self.params_dict['maxRPM'])
        ])
        wheelSpin = [self.normalize(i, 0, self.params_dict['maxWheelSpin']) for i in list(carstate.wheel_velocities)]
        distFromEdge = [self.normalize(i, 0, 200) for i in list(carstate.distances_from_edge)]

        X = np.concatenate((X, wheelSpin, distFromEdge))
        # print(X)

        results = self.esn.predict(X, self.first_step).reshape(2)

        # print(results)

        # gear = results[0]
        steer = results[0]
        accel_brake = results[1]

        print(accel_brake)

        command = Command()

        accel_brake = min(max(-1, accel_brake), 1)
        if accel_brake >= 0:
            command.brake = 0
            command.accelerator = self.normalize_to_int(accel_brake, -1, 1, 0 ,1)

            if carstate.rpm > 8000:
                command.gear = carstate.gear + 1
        else:
            command.brake = self.normalize_to_int(accel_brake, -1, 1, 0, 1)
            command.accelerator = 0

            if carstate.rpm < 2500:
                command.gear = carstate.gear - 1

        # if command.accelerator > 0:
        #     if carstate.rpm > 8000:
        #         command.gear = carstate.gear + 1
        # else:
        #     if carstate.rpm < 2500:
        #         command.gear = carstate.gear - 1

        if not command.gear:
            command.gear = carstate.gear or 1

        # gear = min(max(-1, gear), 1)
        # gear = self.normalize_to_int(gear, -1, 1, -1, 6)
        # command.gear = int(round(gear))

        steer = min(max(-1, steer), 1)
        command.steering = steer

        self.first_step = True

        return command
