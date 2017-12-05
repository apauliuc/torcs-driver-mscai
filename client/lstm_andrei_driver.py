from pytocl.driver import Driver
from pytocl.car import State, Command
import torch
import torch.autograd as autograd
import torch.nn as nn


# noinspection PyPep8Naming,PyMethodMayBeStatic
class LSTMDriver(nn.Module):

    @property
    def range_finder_angles(self):
        """Iterable of 19 fixed range finder directions [deg].

        The values are used once at startup of the client to set the directions
        of range finders. During regular execution, a 19-valued vector of track
        distances in these directions is returned in ``state.State.tracks``.
        """
        return -90, -75, -60, -45, -30, -20, -15, -10, -5, 0, 5, 10, 15, 20, \
               30, 45, 60, 75, 90

    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size):
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

    #         self.out = nn.Sigmoid()

    def init_hidden(self):
        return (autograd.Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_size)),
                autograd.Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_size)))

    def forward(self, x):
        lstm_out, hidden_out = self.lstm(x.view(1, -1), self.hidden)
        linear_out = self.linear(lstm_out)
        self.hidden = (autograd.Variable(hidden_out[0].data), autograd.Variable(hidden_out[1].data))
        #         out = self.out(linear_out)

        return linear_out


class LSTMAndreiDriver(Driver):
    TARGET_SIZE = 3
    INPUT_SIZE = 22
    HIDDEN_SIZE = 22
    NUM_LAYERS = 1
    BATCH_SIZE = 1
    NUM_EPOCHS = 500
    LEARNING_RATE = 0.01

    def __init__(self, logdata=True):
        super().__init__(logdata)

        # def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size):

        self.model = LSTMDriver(self.INPUT_SIZE, self.HIDDEN_SIZE, self.NUM_LAYERS, self.TARGET_SIZE, self.BATCH_SIZE)
        self.model.load_state_dict(torch.load('lstm_params.pt'))

    def drive(self, carstate: State) -> Command:
        x = [
            carstate.speed_x,
            carstate.distance_from_center,
            carstate.angle,
            *carstate.distances_from_edge
        ]

        inputs = autograd.Variable(torch.FloatTensor(x))
        outputs = self.model(inputs)
        acc = outputs.data[0][0][0]
        brake = outputs.data[0][0][1]
        steer = outputs.data[0][0][2]
        command = Command()


        # important to make sure car does not stall
        if acc - brake >= 0:
            acc = 0.4 * (acc - brake)
            brake = 0
        else:
            acc = 0

        if acc > 0:
            if carstate.rpm > 8000:
                command.gear = carstate.gear + 1

        if carstate.rpm < 2500:
                command.gear = carstate.gear - 1

        if not command.gear:
            command.gear = carstate.gear

        command.accelerator = acc
        command.brake = brake
        command.steering = steer

        return command
