from pytocl.driver import Driver
from pytocl.car import State, Command
import models.lstm.v3.steering as steering
import models.lstm.v3.speeding as speeding
import pickle

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
        self.steering_model.train(mode=True)

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

        # print(output.data)

        steer = steering_output.data[0][0][0]
        accelerator = speeding_output.data[0][0][0]
        brake = speeding_output.data[0][0][1]

        command = Command()

        command.accelerator = accelerator
        command.brake = brake
        command.steering = steer

        # if accelerator - brake >= 0:
        #     command.brake = 0
        #     command.accelerator = 1.5 * (accelerator - brake)
        # else:
        #     command.brake = brake
        #     command.accelerator = 0

        # gear shifting
        if accelerator - brake >= 0:
            command.brake = 0
            command.accelerator = accelerator
            if carstate.rpm > 8000:
                command.gear = carstate.gear + 1
            if carstate.rpm < 1000:
                command.gear = carstate.gear - 1
        else:
            command.brake = brake
            command.accelerator = 0
            if carstate.rpm < 2500:
                command.gear = carstate.gear - 1

        if not command.gear:
            command.gear = carstate.gear or 1

        return command

