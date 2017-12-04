import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from copy import deepcopy
import math
import pickle


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
    first = True
    X_full = y_full = None

    for f in training_files:
        # read dataset
        train_ds = pd.read_csv(f, index_col=False, low_memory=False)

        X = train_ds.iloc[:, 3:].values
        y = train_ds.iloc[:, :3].values

        X_full = X if first else np.concatenate((X_full, X))
        y_full = y if first else np.concatenate((y_full, y))

        first = False

    # Fill missing values with mean
    imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imputer = imputer.fit(X_full)
    X_full = imputer.transform(X_full)

    X_train = np.zeros(X_full.shape)
    y_train = np.zeros(y_full.shape)

    if normalization:
        X_train[:, 0] = scale_array(X_full[:, 0], -85, 360)  # speed
        X_train[:, 1] = scale_array(X_full[:, 1], -1, 1)  # track position
        X_train[:, 2] = scale_array(X_full[:, 2], -180, 180)  # angle
        for i in np.arange(3, 22):
            X_train[:, i] = scale_array(X_full[:, i], 0, 200)

        y_train[:, 0] = scale_array(y_full[:, 0], 0, 1, -1, 1)
        y_train[:, 1] = scale_array(y_full[:, 1], 0, 1, -1, 1)
        y_train[:, 2] = scale_array(y_full[:, 2], y_full[:, 2].min(), y_full[:, 2].max())
    else:
        X_train = X_full
        y_train = y_full

    return X_train, y_train
