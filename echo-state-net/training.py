from data_processing import load_training_data
import numpy as np
from ESN import ESN
import pickle

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
        n_inputs=parameters['n_inputs'],
        n_outputs=parameters['n_outputs'],
        n_reservoir=parameters['n_reservoir'],
        spectral_radius=parameters['spectral_radius'],
        sparsity=parameters['sparsity'],
        noise=parameters['noise'],
        teacher_forcing=parameters['teacher_forcing'],
        activation_out=np.tanh,
        inverse_activation_out=safe_arctanh,
        print_state=True,
    )

    global err

    train_predict, err = esn.fit(X_train, y_train)

    if save:
        weights_dict = {
            'W': esn.W,
            'W_in': esn.W_in,
            'W_feedback': esn.W_feedback,
            'W_out': esn.W_out
        }

        save_obj(parameters, 'esn_parameters')
        save_obj(weights_dict, 'esn_weights')
        save_obj(esn.random_state_, 'esn_random_state')
        save_obj(param_dict, 'norm_parameters')


if __name__ == '__main__':
    X, y, param_dict = load_training_data('train_data')

    # print(y.shape)

    params = {
        'n_inputs': 28,
        'n_outputs': 2,
        'n_reservoir': 100,
        'spectral_radius': 0.8,
        'sparsity': 0.2,
        'noise': 0.001,
        'teacher_forcing': True
    }

    train_net(params, X, y, save=True)

    # reservoir_size = [100, 90, 80, 70, 60, 50]
    # spectr_radius = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    # sparse = [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
    #
    # i = 0
    #
    # for r in reservoir_size:
    #     for s in spectr_radius:
    #         for sp in sparse:
    #             params = {
    #                 'n_inputs': 28,
    #                 'n_outputs': 2,
    #                 'n_reservoir': r,
    #                 'spectral_radius': s,
    #                 'sparsity': sp,
    #                 'noise': 0.001,
    #                 'teacher_forcing': True
    #             }
    #
    #             print('{}: {}'.format(i, params))
    #
    #             train_net(params, X, y, save=False)
