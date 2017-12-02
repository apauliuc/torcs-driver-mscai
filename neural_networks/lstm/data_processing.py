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
    return (x - mmin) / (mmax - mmin)


def scale_array(x, mmin=0, mmax=1, new_min=-1, new_max=1):
    x = normalize(x, mmin, mmax)
    return x * (new_max - new_min) + new_min


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


# noinspection PyPep8Naming
def load_training_data(training_files, normalization=True):
    maxSpeedX = -10000
    minSpeedX = 10000
    maxSpeedY = -10000
    minSpeedY = 10000
    maxRPM = 0
    maxWheelSpin = 0
    minWheelSpin = 10000

    for f in training_files:
        train_ds = pd.read_csv(f, header=None)
        X = train_ds.iloc[:, :-4].values

        if X[:, 0].max() > maxSpeedX:
            maxSpeedX = X[:, 0].max()
        if X[:, 0].min() < minSpeedX:
            minSpeedX = X[:, 0].min()

        if X[:, 1].max() > maxSpeedY:
            maxSpeedY = X[:, 1].max()
        if X[:, 1].min() < minSpeedY:
            minSpeedY = X[:, 1].min()

        if X[:, 4].max() > maxRPM:
            maxRPM = X[:, 4].max()

        if X[:, 5:9].max() > maxWheelSpin:
            maxWheelSpin = X[:, 5:9].max()
        if X[:, 5:9].min() > minWheelSpin:
            minWheelSpin = X[:, 5:9].min()

    param_dict = {
        'maxSpeedX': maxSpeedX,
        'minSpeedX': minSpeedX,
        'maxSpeedY': maxSpeedY,
        'minSpeedY': minSpeedY,
        'maxRPM': maxRPM,
        'maxWheelSpin': maxWheelSpin,
        'minWheelSpin': minWheelSpin,
    }

    training_sets = defaultdict(lambda: dict())

    for f in training_files:
        # read dataset
        train_ds = pd.read_csv(f, header=None)

        X = train_ds.iloc[:, :-4].values
        y = train_ds.iloc[:, -4:].values

        # fill missing values with mean
        imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
        imputer = imputer.fit(X)
        X = imputer.transform(X)

        if normalization is not True:
            continue

        # normalize all values for interval [0, 1]
        X[:, 0] = normalize(X[:, 0], minSpeedX, maxSpeedX)  # speedX = range(search min, search max)
        X[:, 1] = normalize(X[:, 1], minSpeedY, maxSpeedY)  # speedY = range(search min, search max)
        X[:, 2] = normalize(X[:, 2], -math.pi, math.pi)  # angle = range(-180, 180)
        X[:, 3] = normalize(np.clip(X[:, 3], -1, 6), -1, 6)  # currentGear = range(-1, 6)
        X[:, 4] = normalize(np.clip(X[:, 4], 0, maxRPM), 0, maxRPM)  # RPM = range(0, search max)
        for i in np.arange(5, 9):
            X[:, i] = normalize(X[:, i], minWheelSpin, maxWheelSpin)  # *wheelSpin = range(0, search max)
        for i in np.arange(9, 28):
            X[:, i] = normalize(np.clip(X[:, i], 0, 200), 0, 200)  # *sensorValues = range(0, 200)
        # y[:, 0] = normalize(y[:, 0], -1, 6)  # gear = range(-1, 6)
        y[:, 0] = normalize(np.clip(y[:, 1], -1, 1), -1, 1)  # steering = range(-1, 1)
        # for acceleration and break, compute their difference and normalize it
        accel_brake = y[:, 2] - y[:, 3]
        y[:, 1] = normalize(accel_brake, -1, 1)  # accelerate-brake = range(-1, 1)
        y = np.delete(y, 3, axis=1)
        y = np.delete(y, 2, axis=1)

        # Create TensorDataset from FloatTensors and save to dictionary
        X_train = torch.from_numpy(X).float()
        y_train = torch.from_numpy(y).float()
        dataset = TensorDataset(X_train, y_train)
        training_sets[f] = dataset

    save_obj(param_dict, 'norm_parameters')

    return training_sets
