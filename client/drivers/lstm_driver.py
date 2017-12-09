from pytocl.driver import Driver
from pytocl.car import State, Command
from models.lstm.v2.model import LSTMDriver, HYPERPARAMS
import pickle

import torch
from torch.autograd import Variable

dtype = torch.FloatTensor


def load_obj(name):
    with open('parameters/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


# noinspection PyPep8Naming,PyMethodMayBeStatic
class Driver_LSTM(Driver):

    def __init__(self, logdata=True):
        super().__init__(logdata)

        self.neural_net = LSTMDriver(
            HYPERPARAMS.INPUT_SIZE,
            HYPERPARAMS.HIDDEN_SIZE,
            HYPERPARAMS.NUM_LAYERS,
            HYPERPARAMS.TARGET_SIZE,
            HYPERPARAMS.BATCH_SIZE
        )

        checkpoint = torch.load('models/lstm/v2/checkpoints/best_checkpoint.pth.tar',
                                map_location=lambda storage, loc: storage)
        self.neural_net.load_state_dict(checkpoint['state_dict'])
        self.neural_net.init_hidden()

    def drive(self, carstate: State) -> Command:
        X = [carstate.speed_x,
             carstate.distance_from_center,
             carstate.angle,
             *carstate.distances_from_edge]

        X = torch.FloatTensor(X)
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
