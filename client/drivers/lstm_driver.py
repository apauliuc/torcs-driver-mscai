from pytocl.driver import Driver
from pytocl.car import State, Command
import models.lstm.v3.steering as steering
import models.lstm.v3.speeding as speeding
import pickle
import json
import os

import torch
from torch.autograd import Variable

FIFO = 'mypipe_random'

two2one = 'two2one_random'
one2two = 'one2two_random'

process = 0

file = open('%s' % one2two, 'w')
file.write("")
file.close()
file = open('%s' % two2one, 'w')
file.write("")
file.close()

try:
    os.mkfifo(FIFO)
    process = 1  # i am in process 1
except OSError as oe:
    process = 2  # i am in process 2
    os.remove(FIFO)


def load_obj(name):
    with open('parameters/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def outside_of_track(car_state: State) -> bool:
    return abs(car_state.distance_from_center) >= 1


def signal(car_state: State):
    if process == 1:
        file_write = open('%s' % one2two, 'w', os.O_NONBLOCK)
    else:
        file_write = open('%s' % two2one, 'w', os.O_NONBLOCK)

    file_write.write(str(car_state.distance_from_start) + "\n")
    file_write.flush()
    file_write.close()


def in_danger(car_state: State):
    if process == 2:
        file_read = open(one2two, 'r')
    else:
        file_read = open(two2one, 'r')

    for other_car_distance_from_start in file_read:
        if other_car_distance_from_start != "":
            if float(other_car_distance_from_start) - float(car_state.distance_from_start) < 100.0 and \
                                    float(other_car_distance_from_start) - float(car_state.distance_from_start) > 0:
                return True
    file_read.close()
    return False


class LSTMDriver(Driver):
    def __init__(self, logdata=True):
        super().__init__(logdata)

        self.already_signaled = False

        self.steering_model = steering.Steering(
            steering.HYPERPARAMS.INPUT_SIZE,
            steering.HYPERPARAMS.LSTM_HIDDEN_SIZE,
            steering.HYPERPARAMS.HIDDEN_LAYER_SIZE,
            steering.HYPERPARAMS.DROPOUT_PROB,
            steering.HYPERPARAMS.NUM_LAYERS,
            steering.HYPERPARAMS.TARGET_SIZE,
            steering.HYPERPARAMS.BATCH_SIZE
        )

        self.speeding_model = speeding.Speeding(
            speeding.HYPERPARAMS.INPUT_SIZE,
            speeding.HYPERPARAMS.LSTM_HIDDEN_SIZE,
            speeding.HYPERPARAMS.HIDDEN_LAYER_SIZE,
            speeding.HYPERPARAMS.DROPOUT_PROB,
            speeding.HYPERPARAMS.NUM_LAYERS,
            speeding.HYPERPARAMS.TARGET_SIZE,
            speeding.HYPERPARAMS.BATCH_SIZE
        )

        steering_params = torch.load('models/lstm/v3/model_parameters/steering/best_model_parameters.tar',
                                     map_location=lambda storage, loc: storage)
        speeding_params = torch.load('models/lstm/v3/model_parameters/speeding/best_model_parameters.tar',
                                     map_location=lambda storage, loc: storage)
        self.steering_model.load_state_dict(steering_params)
        self.speeding_model.load_state_dict(speeding_params)

        self.steering_model.train(mode=False)
        self.speeding_model.train(mode=False)

        self.steering_model.init_hidden()
        self.speeding_model.init_hidden()

    def drive(self, carstate: State) -> Command:
        x = [carstate.speed_x,
             carstate.distance_from_center,
             carstate.angle,
             *carstate.distances_from_edge]

        X = torch.FloatTensor(x)
        inputs = Variable(X.view(1, 1, -1))
        steering_output = self.steering_model(inputs)
        speeding_output = self.speeding_model(inputs)

        steer = steering_output.data[0][0][0]
        accelerator = speeding_output.data[0][0][0]
        brake = speeding_output.data[0][0][1]

        command = Command()

        command.accelerator = accelerator
        command.brake = brake
        command.steering = steer

        # gear shifting
        # gear_params = json.load(open('automatic_gear_params.json'))
        if accelerator - brake >= 0:
            command.accelerator = accelerator
            if carstate.rpm > 8000:
                command.gear = carstate.gear + 1
            if carstate.rpm < 1000:
                command.gear = carstate.gear - 1
        else:
            command.brake = brake
            command.accelerator = 0
            if carstate.rpm < 3000:
                command.gear = carstate.gear - 1

        if not command.gear:
            command.gear = carstate.gear or 1
        # command.gear = automatic_transmission(gear_params, carstate.rpm, carstate.gear)

        try:
            if not self.already_signaled and outside_of_track(carstate):
                signal(carstate)
                print(process, " Outside of track.")
                self.already_signaled = True
            else:
                if in_danger(carstate):
                    print(process, "In danger!")
                    command.accelerator *= 0.8
                    command.brake *= 1.1
        except Exception as e:
            print(e)

        return command


def automatic_transmission(P, rpm, g):
    ng = 1
    if g == 6 and rpm < P['dnsh5rpm']:
        ng = g - 1
    elif g == 5 and rpm < P['dnsh4rpm']:
        ng = g - 1
    elif g == 4 and rpm < P['dnsh3rpm']:
        ng = g - 1
    elif g == 3 and rpm < P['dnsh2rpm']:
        ng = g - 1
    elif g == 2 and rpm < P['dnsh1rpm']:
        ng = g - 1
    elif g == 5 and rpm > P['upsh6rpm']:
        ng = g + 1
    elif g == 4 and rpm > P['upsh5rpm']:
        ng = g + 1
    elif g == 3 and rpm > P['upsh4rpm']:
        ng = g + 1
    elif g == 2 and rpm > P['upsh3rpm']:
        ng = g + 1
    elif g == 1 and rpm > P['upsh2rpm']:
        ng = g + 1
    elif not g:
        ng = 1
    else:
        pass
    return ng
