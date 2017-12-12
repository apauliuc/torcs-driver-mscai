import glob
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset


def normalize(x, mmin, mmax):
    return np.clip((x - mmin) / (mmax - mmin), 0, 1)


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


# noinspection PyPep8Naming
def load_training_data(training_files):
    training_sets = defaultdict(lambda: dict())

    for file in training_files:
        # read dataset
        train_ds = pd.read_csv(file, index_col=False, low_memory=False)

        sensors_data = train_ds.iloc[:, 3:].values
        command_data = train_ds.iloc[:, :3].values

        # Create TensorDataset from FloatTensors and save to dictionary
        X_train = torch.from_numpy(sensors_data).float()
        y_train = torch.from_numpy(command_data).float()
        data_set = TensorDataset(X_train, y_train)
        training_sets[file] = data_set

    return training_sets


if __name__ == '__main__':
    # script_dir = os.path.dirname(__file__)
    # project_dir = os.path.split(os.path.split(script_dir)[0])[0]

    driver = 'default_driver_oval'
    train_data_path = 'data/csv/{}'.format(driver)

    training_files = glob.glob(train_data_path + '/*.csv')

    # training_sets = load_training_data(training_files)
