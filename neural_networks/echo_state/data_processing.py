import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from copy import deepcopy
import math
import pickle


def normalize(x):
    return (x - np.mean(x)) / np.std(x)


def scale_to_0_1(x, mmin, mmax):
    return np.clip((x - mmin) / (mmax - mmin), 0, 1)


def scale_array(x, mmin=0, mmax=1, new_min=-1, new_max=1):
    x = scale_to_0_1(x, mmin, mmax)
    return np.clip(x * (new_max - new_min) + new_min, new_min, new_max)


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

    means = []
    stdevs = []

    if normalization:
        for i in range(X_full.shape[1]):
            means.append(np.mean(X_full[:, i]))
            stdevs.append(np.std(X_full[:, i]))
            X_train[:, i] = normalize(X_full[:, i])

        # X_train[:, 0] = normalize(X_full[:, 0])  # speed
        # X_train[:, 1] = normalize(X_full[:, 1])  # track position
        # X_train[:, 2] = normalize(X_full[:, 2])  # angle
        # for i in np.arange(3, 22):
        #     X_train[:, i] = normalize(X_full[:, i])

        y_train[:, 0] = scale_array(y_full[:, 0], 0, 1)
        y_train[:, 1] = scale_array(y_full[:, 1], 0, 1)
        y_train[:, 2] = scale_array(y_full[:, 2], y_full[:, 2].min(), y_full[:, 2].max())
        # y_train = np.delete(y_train, 2, axis=1)
    else:
        X_train = X_full
        y_train = y_full

    return X_train, y_train, means, stdevs
