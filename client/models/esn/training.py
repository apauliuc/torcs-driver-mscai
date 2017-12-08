from models.esn.data_processing import load_training_data
import numpy as np
import pickle
from models.esn.ESN import ESN

err = 0.0


def safe_arctanh(x):
    if x.ndim == 2:
        for row in np.arange(x.shape[0]):
            x[row, :] = [i - 1e-15 if i == 1 else i + 1e-15 if i == -1 else i for i in x[row, :]]
    elif x.ndim == 1:
        x = [i - 1e-15 if i == 1 else i + 1e-15 if i == -1 else i for i in x]
    return np.arctanh(x)


def save_obj(obj, name):
    with open('parameter_files/' + name + '_%.0f.pkl' % (err * 10000), 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def train_net(parameters, X_train, y_train, save=True):
    esn = ESN(
        n_input=parameters['n_input'],
        n_output=parameters['n_output'],
        n_reservoir=parameters['n_reservoir'],
        spectral_radius=parameters['spectral_radius'],
        leaking_rate=parameters['leaking_rate'],
        reservoir_density=parameters['reservoir_density'],
        out_activation=np.tanh,
        inverse_out_activation=safe_arctanh,
        feedback=parameters['feedback'],
        silent=False
    )

    global err

    err = esn.fit(X_train, y_train)

    if save:
        weights_dict = {
            'W': esn.W,
            'WInput': esn.WInput,
            'WFeedback': esn.WFeedback,
            'WOut': esn.WOut
        }

        save_obj(parameters, 'esn_parameters')
        save_obj(weights_dict, 'esn_weights')
        save_obj(esn.random_state_, 'esn_random_state')
        # save_obj(param_dict, 'norm_parameters')


if __name__ == '__main__':
    X, y, param_dict = load_training_data('train_data')

    params = {
        'n_input': 28,
        'n_output': 3,
        'n_reservoir': 75,
        'spectral_radius': 0.9,
        'leaking_rate': 0.75,
        'reservoir_density': 0.8,
        'feedback': True,
    }

    train_net(params, X, y, save=True)


