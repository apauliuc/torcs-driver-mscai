from pytocl.driver import Driver
from pytocl.car import State, Command
import models.lstm.v3.steering as steering
import models.lstm.v3.speeding as speeding
import pickle
import json

import torch
from torch.autograd import Variable


def load_obj(name):
    with open('parameters/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


class LSTMDriver(Driver):

    def __init__(self, logdata=True):
        super().__init__(logdata)

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

        steering_checkpoint = torch.load('models/lstm/v3/split_checkpoints/steering/best_checkpoint.tar',
                                         map_location=lambda storage, loc: storage)
        speeding_checkpoint = torch.load('models/lstm/v3/split_checkpoints/speeding/best_checkpoint.tar',
                                         map_location=lambda storage, loc: storage)
        self.steering_model.load_state_dict(steering_checkpoint['state_dict'])
        self.speeding_model.load_state_dict(speeding_checkpoint['state_dict'])

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
        # command.gear = automatic_transmission(gear_params, carstate.rpm, carstate.gear, carstate.speed_x)

        return command


def automatic_transmission(P, rpm, g, sx):
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
