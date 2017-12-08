from pytocl.driver import Driver
from pytocl.car import State, Command
from drivers.esn_driver import ESNDriver
import numpy as np
import pickle


def load_obj(name):
    with open('parameters/' + name + '_' + '2782' + '.pkl', 'rb') as f:
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

        self.esn = ESNDriver(
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

        self.first_step = False

    def normalize(self, x, mmin, mmax):
        norm = (x - mmin) / (mmax - mmin)
        return np.clip(norm, mmin, mmax)

    def invert_normalize(self, x, mmin, mmax):
        x = np.clip(x, -1, 1)
        return x * (mmax - mmin) + mmin

    def normalize_to_int(self, x, mmin, mmax, a=-1, b=1):
        return (b - a) * (x - mmin) / (mmax - mmin) + a

    def drive(self, carstate: State) -> Command:

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

        results = self.esn.predict(X, self.first_step).reshape(self.net_p['n_output'])

        steer, acc, brake = results

        command = Command()

        acc = self.normalize(np.clip(acc, -1, 1), -1, 1)
        brake = self.normalize(np.clip(brake, -1, 1), -1, 1)

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

        command.steering = np.clip(steer, -1, 1)

        self.first_step = True

        return command
