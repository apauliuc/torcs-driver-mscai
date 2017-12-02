from pytocl.driver import Driver
from pytocl.car import State, Command
from ESN import ESN
import numpy as np
import pickle
import math


def load_obj(name):
    with open('parameters/' + name + '_' + '2216' + '.pkl', 'rb') as f:
        return pickle.load(f)


def safe_arctanh(x):
    if x.ndim == 2:
        for row in np.arange(x.shape[0]):
            x[row, :] = [i - 1e-15 if i == 1 else i + 1e-15 if i == -1 else i for i in x[row, :]]
    elif x.ndim == 1:
        x = [i - 1e-15 if i == 1 else i + 1e-15 if i == -1 else i for i in x]
    return np.arctanh(x)


# noinspection PyPep8Naming,PyMethodMayBeStatic
class MyDriver(Driver):

    def __init__(self, logdata=True):
        super().__init__(logdata)

        self.net_p = load_obj('esn_parameters')

        self.esn = ESN(
            n_input=self.net_p['n_input'],
            n_reservoir=self.net_p['n_reservoir'],
            n_output=self.net_p['n_output'],
            spectral_radius=self.net_p['spectral_radius'],
            leaking_rate=self.net_p['leaking_rate'],
            reservoir_density=self.net_p['reservoir_density'],
            out_activation=np.tanh,
            inverse_out_activation=safe_arctanh,
            feedback=False,
            silent=True
        )

        self.esn.random_state_ = load_obj('esn_random_state')
        weights = load_obj('esn_weights')
        self.esn.W = weights['W']
        self.esn.WInput = weights['WInput']
        self.esn.WFeedback = weights['WFeedback']
        self.esn.WOut = weights['WOut']

        self.params_dict = load_obj('norm_parameters')

        self.first_step = False

    def normalize(self, x, mmin, mmax):
        norm = (x - mmin) / (mmax - mmin)
        return np.clip(norm, mmin, mmax)

    def invert_normalize(self, x, mmin, mmax):
        x = np.clip(x, -1, 1)
        return x * (mmax - mmin) + mmin

    def scale_array(self, x, mmin=0, mmax=1, new_min=-1, new_max=1):
        x = self.normalize(x, mmin, mmax)
        return x * (new_max - new_min) + new_min

    def drive(self, carstate: State) -> Command:
        X = np.array([
            self.scale_array(carstate.speed_x, self.params_dict['minSpeedX'], self.params_dict['maxSpeedX']),
            self.scale_array(carstate.speed_y, self.params_dict['minSpeedY'], self.params_dict['maxSpeedY']),
            self.scale_array(carstate.angle, -math.pi, math.pi),
            self.scale_array(carstate.gear, -1, 6),
            self.scale_array(carstate.rpm, 0, self.params_dict['maxRPM'])
        ])
        wheels = list(carstate.wheel_velocities)
        wheelSpin = np.zeros(len(wheels))
        for i in np.arange(len(wheels)):
            wheelSpin[i] = self.scale_array(wheels[i],
                                            self.params_dict['minWheelSpin'][i],
                                            self.params_dict['maxWheelSpin'][i])

        distFromEdge = [self.scale_array(i, self.params_dict['minDistFromEdge'], 200)
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

        X = np.concatenate((X, wheelSpin, distFromEdge))

        results = self.esn.predict(X, self.first_step).reshape(self.net_p['n_output'])

        steer, acc, brake = results

        command = Command()

        print(acc)

        acc = self.normalize(np.clip(acc, -1, 1), -1, 1)
        brake = self.normalize(np.clip(brake, -1, 1), -1, 1)

        # print(acc)
        # print(brake)
        # print(steer)

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

        # print(steer)
        command.steering = np.clip(steer, -1, 1)

        self.first_step = True

        return command
