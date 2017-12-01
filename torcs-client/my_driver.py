from pytocl.driver import Driver
from pytocl.car import State, Command
from finalESN import ESN

import numpy as np
import pickle

def load_obj(name):
    with open('parameters/' + name + '_' + '3843' + '.pkl', 'rb') as f:
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
            n_input=self.net_p['n_inputs'],
            n_reservoir=self.net_p['n_reservoir'],
            n_output=self.net_p['n_outputs'],
            spectral_radius=self.net_p['spectral_radius'],
            leaking_rate=self.net_p['leaking_rate'],
            reservoir_density=self.net_p['reservoir_density'],
            out_activation=np.tanh,
            inverse_out_activation=safe_arctanh,
            feedback=self.net_p['feedback'],
            silent=False
        )

        self.esn.random_state_ = load_obj('esn_random_state')
        weights = load_obj('esn_weights')
        self.esn.W = weights['W']
        self.esn.WIn = weights['W_in']
        self.esn.WFeedback = weights['W_feedback']
        self.esn.WOut = weights['W_out']

        # self.esn = ESN(
        #     n_inputs=self.net_p['n_inputs'],
        #     n_outputs=self.net_p['n_outputs'],
        #     n_reservoir=self.net_p['n_reservoir'],
        #     spectral_radius=self.net_p['spectral_radius'],
        #     sparsity=self.net_p['sparsity'],
        #     noise=self.net_p['noise'],
        #     teacher_forcing=self.net_p['teacher_forcing'],
        #     activation_out=np.tanh,
        #     inverse_activation_out=safe_arctanh,
        #     print_state=False
        # )
        #
        # self.esn.random_state_ = load_obj('esn_random_state')
        # weights = load_obj('esn_weights')
        # self.esn.W = weights['W']
        # self.esn.W_in = weights['W_in']
        # self.esn.W_feedback = weights['W_feedback']
        # self.esn.W_out = weights['W_out']

        # self.params_dict = load_obj('norm_parameters')

        self.first_step = False

    def normalize(self, x, mmin, mmax):
        norm = (x - mmin)/(mmax-mmin)
        return np.clip(norm, -1, 1)

    def invert_normalize(self, x, mmin, mmax):
        x = np.clip(x, -1, 1)
        return x * (mmax - mmin) + mmin

    def normalize_to_int(self, x, mmin, mmax, a=-1, b=1):
        return (b - a) * (x - mmin) / (mmax - mmin) + a

    def drive(self, carstate: State) -> Command:
        # X = np.array([
        #     self.normalize(carstate.speed_x, self.params_dict['minSpeedX'], self.params_dict['maxSpeedX']),
        #     self.normalize(carstate.speed_y, self.params_dict['minSpeedY'], self.params_dict['maxSpeedY']),
        #     self.normalize(carstate.angle, -180, 180),
        #     self.normalize(carstate.gear, -1, 6),
        #     self.normalize(carstate.rpm, 0, self.params_dict['maxRPM'])
        # ])
        # wheelSpin = [self.normalize(i, 0, self.params_dict['maxWheelSpin']) for i in list(carstate.wheel_velocities)]
        # distFromEdge = [self.normalize(i, 0, 200) for i in list(carstate.distances_from_edge)]

        X = np.array([
            carstate.speed_x,
            carstate.speed_y,
            carstate.angle,
            carstate.gear,
            carstate.rpm
        ])
        wheelSpin = [i for i in list(carstate.wheel_velocities)]
        distFromEdge = [i for i in list(carstate.distances_from_edge)]

        X = np.concatenate((X, wheelSpin, distFromEdge))

        results = self.esn.predict(X, self.first_step).reshape(self.net_p['n_outputs'])

        steer, acc_brake = results

        # gear = results[0]
        # steer = results[0]
        # acc_brake = results[1]

        # print(steer)

        command = Command()

        acc_brake = np.clip(acc_brake, -1, 1)

        # gear = self.normalize_to_int(gear, -1, 1, -1, 6)
        # gear = int(round(gear))

        # print('{} {}'.format(gear, steer))

        if acc_brake > 0:
            command.brake = 0
            command.accelerator = acc_brake

            if carstate.rpm > 8000:
                command.gear = carstate.gear + 1
        else:
            command.brake = np.abs(acc_brake)
            command.accelerator = 0

            if carstate.rpm < 2500:
                command.gear = carstate.gear - 1

        if not command.gear:
            command.gear = carstate.gear or 1

        # print(steer)
        command.steering = np.clip(steer, -1, 1)

        self.first_step = True

        return command
