import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import Imputer
from copy import deepcopy

import math
from collections import defaultdict
from torch.utils.data import TensorDataset
import torch


def normalize(x, mmin, mmax):
    return np.clip((x - mmin) / (mmax - mmin), 0, 1)


def scale_array(x, mmin=0, mmax=1, new_min=-1, new_max=1):
    x = normalize(x, mmin, mmax)
    return np.clip(x * (new_max - new_min) + new_min, new_min, new_max)


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


# noinspection PyPep8Naming
def load_training_data(training_files, normalization=True):
    training_sets = defaultdict(lambda: dict())

    for f in training_files:
        # read dataset
        train_ds = pd.read_csv(f, index_col=False, low_memory=False)

        X = train_ds.iloc[:, 3:].values
        y = train_ds.iloc[:, :3].values

        # fill missing values with mean
        imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
        imputer = imputer.fit(X)
        X = imputer.transform(X)

        if normalization is not True:
            continue

        # normalize all values for interval [0, 1]
        X[:, 0] = normalize(X[:, 0], -85, 360)  # speed
        X[:, 1] = normalize(X[:, 1], -1, 1)  # track position
        X[:, 2] = normalize(X[:, 2], -180, 180)  # angle
        for i in np.arange(3, 22):
            X[:, i] = normalize(X[:, i], 0, 200)

        # y[:, 0] = normalize(y[:, 0], 0, 1)  # acceleration
        # y[:, 1] = normalize(y[:, 1], 0, 1)  # brake
        # y[:, 2] = normalize(y[:, 2], y[:, 2].min(), y[:, 2].max())

        # Create TensorDataset from FloatTensors and save to dictionary
        X_train = torch.from_numpy(X).float()
        y_train = torch.from_numpy(y).float()
        dataset = TensorDataset(X_train, y_train)
        training_sets[f] = dataset

    return training_sets
