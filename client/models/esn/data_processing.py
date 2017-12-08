import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer


def normalize(x, mmin, mmax):
    return (x - mmin) / (mmax - mmin)


def scale_array(x, mmin=0, mmax=1, new_min=-1, new_max=1):
    x = normalize(x, mmin, mmax)
    return x * (new_max - new_min) + new_min


# noinspection PyPep8Naming
def load_training_data(folder, normalization=True):
    training_files = glob.glob(folder + '/bot*.csv')

    first = True
    X_full = y_full = None

    for f in training_files:
        # read dataset
        train_ds = pd.read_csv(f, header=None)

        X = train_ds.iloc[:, :-4].values
        y = train_ds.iloc[:, -4:].values

        X_full = X if first else np.concatenate((X_full, X))
        y_full = y if first else np.concatenate((y_full, y))

        first = False

    # Fill missing values with mean
    imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imputer = imputer.fit(X_full)
    X_full = imputer.transform(X_full)

    # Min and max values for specific parameters
    maxSpeedX = X_full[:, 0].max()
    minSpeedX = X_full[:, 0].min()
    maxSpeedY = X_full[:, 1].max()
    minSpeedY = X_full[:, 1].min()
    maxRPM = X_full[:, 4].max()
    maxWheelSpin = X_full[:, 5:9].max()
    minWheelSpin = X_full[:, 5:9].min()
    minDistFromEdge = X_full[:, 9:].min()

    param_dict = {
        'maxSpeedX': maxSpeedX,
        'minSpeedX': minSpeedX,
        'maxSpeedY': maxSpeedY,
        'minSpeedY': minSpeedY,
        'maxRPM': maxRPM,
        'maxWheelSpin': maxWheelSpin,
        'minWheelSpin': minWheelSpin,
        'minDistFromEdge': minDistFromEdge
    }

    X_train = X_full
    y_train = np.zeros(y_full.shape)

    if normalization:
        # # speedX = range(search min, search max)
        # X_train[:, 0] = normalize_to_int(X_full[:, 0], minSpeedX, maxSpeedX)
        # # speedY = range(search min, search max)
        # X_train[:, 1] = normalize_to_int(X_full[:, 1], minSpeedY, maxSpeedY)
        # # angle = range(-180, 180)
        # X_train[:, 2] = normalize_to_int(X_full[:, 2], -180, 180)
        # # currentGear = range(-1, 6)
        # X_train[:, 3] = normalize_to_int(X_full[:, 3], -1, 6)
        # # RPM = range(0, search max)
        # X_train[:, 4] = normalize_to_int(X_full[:, 4], 0, maxRPM)
        # # *wheelSpin = range(0, search max)
        # for i in np.arange(5, 9):
        #     X_train[:, i] = normalize_to_int(X_full[:, i], minWheelSpin, maxWheelSpin)
        # # *sensorValues = range(0, 200)
        # for i in np.arange(9, 28):
        #     X_train[:, i] = normalize_to_int(X_full[:, i], minDistFromEdge, 200)
        #

        # # gear = range(-1, 6)
        # gears = y_full[:, 0]
        # gears[gears > 6] = 6
        # y_train[:, 0] = normalize_to_int(gears, -1, 6)

        # steering = range(-1, 1)
        y_train[:, 0] = np.clip(y_full[:, 1], -1, 1)
        # y_train[:,1] = normalize2(y_full[:,1], -1, 1)  # steering = range(-1, 1)

        y_train[:, 1] = scale_array(np.clip(y_full[:, 2], 0, 1), new_min=-1, new_max=1)
        y_train[:, 2] = scale_array(np.clip(y_full[:, 3], 0, 1), new_min=-1, new_max=1)

        # # accelerate-brake = range(-1, 1)
        # # for acceleration and break, compute their difference and normalize it
        # accel_brake = y_full[:, 2] - y_full[:, 3]
        # y_train[:, 1] = normalize_to_int(accel_brake, -1, 1)
        # # y_train[:,2] = normalize_to_int(accel_brake, -1, 1)
    else:
        y_train = y_full

    y_train = np.delete(y_train, 3, axis=1)
    # y_train = np.delete(y_train, 2, axis=1)

    return X_train, y_train, param_dict
