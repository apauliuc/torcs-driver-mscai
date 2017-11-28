import numpy as np


# noinspection PyAttributeOutsideInit,PyPep8Naming
class ESN:
    """Simple Echo State Network implementation"""
    def __init__(self, n_inputs, n_outputs, n_reservoir=50, spectral_radius=0.95,
                 sparsity=0, noise=0.001, teacher_forcing=True,
                 activation_out=lambda x: x, inverse_activation_out=lambda x: x,
                 random_state=None, print_state=True):
        # save parameters
        self.n_inputs = n_inputs
        self.n_reservoir = n_reservoir
        self.n_outputs = n_outputs
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.noise = noise

        self.activation_out = activation_out
        self.inverse_activation_out = inverse_activation_out
        self.random_state = random_state

        # random_state might be seed, RandomState object or None
        if isinstance(random_state, np.random.RandomState):
            self.random_state_ = random_state
        elif random_state:
            try:
                self.random_state_ = np.random.RandomState(random_state)
            except TypeError as e:
                raise Exception("Invalid seed: " + str(e))
        else:
            self.random_state_ = np.random.RandomState()

        self.teacher_forcing = teacher_forcing
        self.print_state = print_state

        # initialise recurrent weights
        self.init_weights()

    def init_weights(self):
        """
        Initialise random weights
        """
        # random matrix centered around zero
        W = self.random_state_.rand(self.n_reservoir, self.n_reservoir) - 0.5
        # delete fraction of connections given by self.sparsity
        W[self.random_state_.rand(*W.shape) < self.sparsity] = 0
        # compute spectral radius of weights
        radius = np.max(np.abs(np.linalg.eigvals(W)))
        # rescale W
        self.W = W * (self.spectral_radius / radius)

        # initialize input and feedback weights
        self.W_in = self.random_state_.rand(self.n_reservoir, self.n_inputs) * 2 - 1
        self.W_feedback = self.random_state_.rand(self.n_reservoir, self.n_outputs) * 2 - 1

    def _update(self, state, in_data, out_data):
        if self.teacher_forcing:
            preactiv = (self.W.dot(state) +
                        self.W_in.dot(in_data) +
                        self.W_feedback.dot(out_data))
        else:
            preactiv = (self.W.dot(state) +
                        self.W_in.dot(in_data))

        return (self.activation_out(preactiv) +
                self.noise * (self.random_state_.rand(1, self.n_reservoir) - 0.5))

    def fit(self, inputs, outputs, inspect=False):
        """
        Train the echos state network based on input and output data

        Args:
            inputs (array of dimension (N x n_inputs)): input data
            outputs (array of dimension (N x n_outputs)): output data
            inspect (bool): visualise data

        Returns:
            network output on training data, based on trained weights
        """
        if inputs.ndim < 2:
            inputs = np.reshape(inputs, (len(inputs), -1))
        if outputs.ndim < 2:
            outputs = np.reshape(outputs, (len(outputs), -1))

        # compute reservoir states
        if self.print_state:
            print("computing states...")
        states = np.zeros((inputs.shape[0], self.n_reservoir))
        for i in range(1, inputs.shape[0]):
            states[i, :] = self._update(states[i - 1],
                                        inputs[i, :],
                                        outputs[i - 1, :])

        # fit the weights to states and output
        if self.print_state:
            print("fitting...")
        # compute washout time T_0 and skip these data points
        washout_time = min(int(inputs.shape[1] / 10), 100)
        M = np.hstack((states, inputs))
        M_inv = np.linalg.pinv(M[washout_time:, :])
        self.W_out = M_inv.dot(self.inverse_activation_out(outputs[washout_time:, :])).T

        self.last_state = states[-1, :]
        self.last_input = inputs[-1, :]
        self.last_output = outputs[-1, :]

        if inspect:
            from matplotlib import pyplot as plt
            plt.figure(figsize=(states.shape[0] * 0.0025, states.shape[1] * 0.01))
            plt.imshow(M.T, aspect='auto', interpolation='nearest')
            plt.colorbar()

        # compute training error on computed weights
        if self.print_state:
            print("training error:")
        pred_train = self.activation_out(M.dot(self.W_out.T))
        pred_err = np.sqrt(np.mean((pred_train - outputs) ** 2))
        if self.print_state:
            print(pred_err)
        return pred_train, pred_err

    def predict(self, inputs, continuation_train=False):
        """
        Compute new prediction from given input and new weights

        Args:
            inputs (array of dimension (N_samples x n_inputs)): input data
            continuation_train (bool): if True, start network from last training state

        Returns:
            array of output activations
        """
        if inputs.ndim < 2:
            inputs = np.reshape(inputs, (len(inputs), -1))
        n_samples = inputs.shape[0]

        last_state = self.last_state if continuation_train else np.zeros(self.n_reservoir)
        last_input = self.last_input if continuation_train else np.zeros(self.n_inputs)
        last_output = self.last_output if continuation_train else np.zeros(self.n_outputs)

        inputs = np.vstack([last_input, inputs])
        states = np.vstack([last_state, np.zeros((n_samples, self.n_reservoir))])
        outputs = np.vstack([last_output, np.zeros((n_samples, self.n_outputs))])

        for i in range(n_samples):
            states[i + 1, :] = self._update(states[i, :], inputs[i + 1, :], outputs[i, :])
            outputs[i + 1, :] = self.activation_out(
                self.W_out.dot(np.concatenate([states[i + 1, :], inputs[i + 1, :]]))
            )

        return self.activation_out(outputs[1:])
