import numpy as np
from sklearn.utils import check_random_state


# noinspection PyPep8Naming,PyAttributeOutsideInit
class ESN(object):
    def __init__(self, n_input, n_output, n_reservoir=50,
                 spectral_radius=1.0, leaking_rate=0.1,
                 reservoir_density=0.8, random_state=None,
                 out_activation=lambda x: x, inverse_out_activation=lambda x: x,
                 feedback=True, silent=True):
        self.n_input = n_input
        self.n_reservoir = n_reservoir
        self.n_output = n_output

        self._spectral_radius = spectral_radius
        self._reservoir_density = reservoir_density
        self._leaking_rate = leaking_rate
        self._feedback = feedback
        self._silent = silent

        self.out_activation = out_activation
        self.inverse_out_activation = inverse_out_activation

        self.random_state_ = check_random_state(random_state)

        self._create_reservoir()

    def _create_reservoir(self):
        # weight matrix with values [-0.5, 0.5]
        self.W = self.random_state_.rand(self.n_reservoir, self.n_reservoir) - 0.5

        # make weight matrix sparse based on reservoir density
        if self._reservoir_density is not None:
            mask = self.random_state_.rand(*self.W.shape) > self._reservoir_density
            self.W[mask] *= 0.0

        # rescale weight matrix to wanted spectral radius
        if self._spectral_radius is not None:
            radius = np.max(np.abs(np.linalg.eigvals(self.W)))
            self.W = self.W * (self._spectral_radius / radius)

        # random input weight matrix
        self.WInput = self.random_state_.rand(self.n_reservoir, 1 + self.n_input) - 0.5

        # random feedback weight matrix
        self.WFeedback = self.random_state_.rand(self.n_reservoir, 1 + self.n_output) - 0.5

    def _update(self, in_data, state, out_data):
        # feed the output into preactivation if feedback is set to True
        if self._feedback:
            preactiv = (np.dot(self.WInput, in_data) +
                        np.dot(self.W, state) +
                        np.dot(self.WFeedback, out_data))
        else:
            preactiv = (np.dot(self.WInput, in_data) +
                        np.dot(self.W, state))

        # leaking rate specifies how much previous state influences the output
        return (1 - self._leaking_rate) * state + self._leaking_rate * self.out_activation(preactiv)

    def fit(self, X, y):
        # reshape data if not 2dim
        if X.ndim < 2:
            X = np.reshape(X, (len(X), -1))
        if y.ndim < 2:
            y = np.reshape(y, (len(y), -1))

        # compute the states matrix
        if not self._silent:
            print("computing states...")

        # washout time to not feed the output into the computation for first few inputs
        washout_t = min(int(X.shape[0] / 10), 100)
        self._feedback = False

        states = np.zeros((X.shape[0], self.n_reservoir))
        for i in np.arange(X.shape[0] - 1):
            if self._feedback is False and i > washout_t:
                self._feedback = True

            states[i + 1, :] = self._update(np.hstack((1, X[i + 1, :])),
                                            states[i, :],
                                            np.hstack((1, y[i, :])))

        # time to fit the output to the weight matrix
        if not self._silent:
            print("fitting...")

        # create M matrix
        M = np.hstack((states[washout_t:, :], X[washout_t:, :], y[washout_t - 1:-1, :]))
        # invert M
        M_inv = np.linalg.pinv(M)
        # create T matrix
        T_matrix = self.inverse_out_activation(y[washout_t:, :])
        # compute W out
        self.WOut = np.dot(M_inv, T_matrix).T

        # get ESN performance indicator based on training data and the newly computed WOut matrix
        M = np.hstack((states, X, y))
        pred_train = self.out_activation(np.dot(M, self.WOut.T))
        pred_err = np.sqrt(np.mean((pred_train - y) ** 2))
        if not self._silent:
            print("training error:")
            print(pred_err)

        return pred_err

    def predict(self, inputs, continuation_train=False):
        # reshape data
        if inputs.ndim < 2:
            inputs = np.reshape(inputs, (1, -1))
        n_samples = inputs.shape[0]

        # check for previous input, state and output, otherwise 0
        last_input = self.last_input if continuation_train else np.zeros(self.n_input).reshape(1, -1)
        last_state = self.last_state if continuation_train else np.zeros(self.n_reservoir)
        last_output = self.last_output if continuation_train else np.zeros(self.n_output)

        # matrices with first row for previous data and second for the new, to be computed, data
        inputs = np.vstack([last_input, inputs])
        states = np.vstack([last_state, np.zeros((n_samples, self.n_reservoir))])
        outputs = np.vstack([last_output, np.zeros((n_samples, self.n_output))])

        # for each row in input, compute the new state and its output
        for i in range(n_samples):
            states[i + 1, :] = self._update(np.hstack((1, inputs[i + 1, :])),
                                            states[i, :],
                                            np.hstack((1, outputs[i, :])))
            outputs[i + 1, :] = self.out_activation(
                np.dot(self.WOut, np.concatenate([states[i + 1, :], inputs[i + 1, :], outputs[i, :]]))
            )

        # remember last input state and output
        self.last_input = inputs[-1, :]
        self.last_state = states[-1, :]
        self.last_output = outputs[-1, :]

        return outputs[-1:]
