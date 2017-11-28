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
    with open('parameter_files/' + name + '_%.0f.pkl' %(err*10000), 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    X_train, y_train, param_dict = load_training_data('train_data')

    echo_net = ESN(
        n_inputs=28,
        n_outputs=3,
        n_reservoir=100,
        spectral_radius=0.9,
        sparsity=0.2,
        noise=0.001,
        teacher_forcing=True,
        activation_out=np.tanh,
        inverse_activation_out=safe_arctanh,
        print_state=True,
        # random_state=10
    )

    train_predict, err = echo_net.fit(X_train, y_train)

    last_data = {
        'last_input': echo_net.last_input,
        'last_state': echo_net.last_state,
        'last_output': echo_net.last_output
    }
    weights_dict = {
        'W': echo_net.W,
        'W_in': echo_net.W_in,
        'W_feedback': echo_net.W_feedback,
        'W_out': echo_net.W_out
    }

    save_obj(last_data, 'esn_last_data')
    save_obj(last_data, 'esn_weights')
    save_obj(echo_net.random_state_, 'esn_random_state')
    save_obj(echo_net, 'esn_full')
    save_obj(param_dict, 'norm_parameters')

    # # Save to npy with numpy arrays
    # np.save('esn_last_data', np.array([echo_net.last_input,
    #                                    echo_net.last_state,
    #                                    echo_net.last_output]), allow_pickle=True)
    # np.save('esn_W', np.array([echo_net.W,
    #                            echo_net.W_in,
    #                            echo_net.W_feedback,
    #                            echo_net.W_out]), allow_pickle=True)
