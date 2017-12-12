import os
import glob
import copy
import random

import pickle
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

from LSTMDriver import LSTMDriver
from data_processing import load_training_data


def load_obj(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def save_obj(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


# script_dir = os.path.dirname(__file__)
# project_dir = os.path.split(os.path.split(script_dir)[0])[0]

drivers = ['default_driver_dirt', 'default_driver_full', 'default_driver_oval', 'default_driver_road']
training_files = []
testing_files = []
for driver in drivers[2:3]:
    train_data_path = 'data/csv/{}'.format(driver)
    training_files.extend(glob.glob(train_data_path + '/*.csv')[:-1])
    testing_files.extend(glob.glob(train_data_path + '/*.csv')[-1:])

# FILES
TRAINING_FILENAMES = training_files
TEST_FILENAMES = testing_files

INITIAL_LSTM_PARAMETERS_RELATIVE_PATH = "parameters/evolution/rnn_params.pkl"       # from here I take the lstm hypermarams(if not found, I resort to manual settings.
INITIAL_LSTM_WEIGHTS_RELATIVE_PATH = "parameters/evolution/best_model_state.pth.tar"    # from here i take the seed for the lstm model.
LSTM_EVOLUTION_WEIGHTS_PATH = "parameters/evolution/during_climbing"        # here i save the best model after evolutions..


try:
    params = load_obj(INITIAL_LSTM_PARAMETERS_RELATIVE_PATH)
except IOError:
    params = {
        'input': 22,
        'hidden': 75,
        'output': 3,
        'layers': 2,
    }
    save_obj(params, INITIAL_LSTM_PARAMETERS_RELATIVE_PATH)

INPUT_SIZE = params['input']
HIDDEN_SIZE = params['hidden']
OUTPUT_SIZE = params['output']
NUM_LAYERS = params['layers']

MILESTONES = [20, 30, 40]
GAMMA = 0.1

# TRAIN & TEST
BATCH_SIZE = 1

# TRAIN
NUM_EPOCHS = 25
LEARNING_RATE = 1e-3

# EVOLUTION
NO_GENERATIONS = 10
POPULATION_SIZE = 16
INVERS_MUTATION_MAGNITUDE = 1e-2

if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor


def train_model(model):
    model.train(True)

    training_sets = load_training_data(TRAINING_FILENAMES)

    if torch.cuda.is_available():
        criterion = nn.MSELoss().cuda()
    else:
        criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(0, NUM_EPOCHS):
        epoch_error = 0
        print('\tEpoch [%d/%d] error: ' % (epoch + 1, NUM_EPOCHS), end='', flush=True)
        for file in TRAINING_FILENAMES:
            model.hidden = model.init_hidden()

            data = Variable(training_sets[file].data_tensor.view(-1, 1, INPUT_SIZE)).type(dtype)
            target = Variable(training_sets[file].target_tensor).type(dtype)

            optimizer.zero_grad()
            prediction = model(data)
            loss = criterion(prediction, target)
            loss.backward()
            optimizer.step()

            epoch_error += loss.data[0]

        print('{}'.format(epoch_error / len(TRAINING_FILENAMES)))

    return model


def train_models(models):
    return [train_model(model) for model in models]


def score_model(model):
    # Behaves differently when testing.
    model.train(True)
    model.eval()

    testing_sets = load_training_data(TEST_FILENAMES)

    if torch.cuda.is_available():
        criterion = nn.MSELoss().cuda()
    else:
        criterion = nn.MSELoss()

    error = 0
    for file in TEST_FILENAMES:
        model.hidden = model.init_hidden()

        data = Variable(testing_sets[file].data_tensor.view(-1, 1, INPUT_SIZE)).type(dtype)
        target = Variable(testing_sets[file].target_tensor).type(dtype)

        loss = criterion(model(data), target)

        error += loss.data[0]

    # print('\tScore: {}'.format(error))
    return error


def score_models(models):
    print("\t Scoring {} models..".format(str(len(models))))
    scored_models = [(model, score_model(model)) for model in models]
    scores = sorted([score for _, score in scored_models])
    print("\t Best score: {}\n\t Worst score:{}".format(scores[0], scores[-1]))
    return scored_models


def get_small_numbers(shape):
    small_numbers = INVERS_MUTATION_MAGNITUDE * np.ones(shape)
    no_elements = np.prod(shape)
    flat_small_numbers = small_numbers.reshape(no_elements)
    small_numbers = np.array([random.random() * number for number in flat_small_numbers]).reshape(shape)
    return small_numbers


def change_weights(matrix, number):
    matrix = np.array(matrix, dtype=np.float)

    shape = matrix.shape
    no_elements = np.prod(shape)

    if number > no_elements:
        number = no_elements;

    flat_matrix = matrix.reshape(no_elements)

    indices = np.random.choice(np.arange(len(flat_matrix)), number, replace=False)
    small_numbers = get_small_numbers(number)

    for j, i in enumerate(indices):
        if random.random() < 0.5:
            flat_matrix[i] += small_numbers[j]
        else:
            flat_matrix[i] -= small_numbers[j]

    return flat_matrix.reshape(shape)


def add_node_mutation(model):
    model.cpu()
    weights = model.state_dict()
    new_hidden = model.hidden_size + 1
    mutated_model = LSTMDriver(INPUT_SIZE, new_hidden, NUM_LAYERS, OUTPUT_SIZE, BATCH_SIZE)
    for key in weights.keys():
        if key.startswith('lstm'):  # changing lstm layer weights.
            shape = list(weights[key].shape)
            elems = shape[0]
            shape[0] = 1
            shape = tuple(shape)

            weights[key] = torch.from_numpy(
                np.insert(weights[key].numpy(), [x for x in range(elems // 4, elems + 1, elems // 4)],
                          get_small_numbers(shape)[0], axis=0))

            if key[-1] != '0' and 'weight_ih_' in key:
                weights[key] = torch.from_numpy(
                    np.insert(weights[key].numpy(), weights[key].shape[1],
                              get_small_numbers((shape[1], shape[0]))[0], axis=1))

            if key.startswith("lstm.weight_hh_l"):
                weights[key] = torch.from_numpy(
                    np.insert(weights[key].numpy(), weights[key].shape[1],
                              get_small_numbers((shape[1], shape[0]))[0], axis=1))
        elif key == 'linear.weight':
            weights[key] = torch.from_numpy(
                np.insert(weights[key].numpy(), weights[key].shape[1],
                          get_small_numbers((weights[key].shape[1], 1))[0], axis=1))
        else:
            pass

    mutated_model.load_state_dict(weights)

    if torch.cuda.is_available():
        mutated_model.cuda()

    return mutated_model


def change_weights_mutation(model, number=None):
    if number is None:
        mutation_type = random.random()
        if mutation_type < 0.4:
            return change_weights_mutation(model, number=10)
        elif mutation_type >= 0.6:
            return change_weights_mutation(model, number=15)
        else:
            return change_weights_mutation(model, number=20)
    else:
        model.cpu()
        for key in model.state_dict().keys():
            if key.startswith('lstm'):  # changing lstm layer weights.
                shape = model.state_dict()[key].shape
                hidden_size = shape[0] // 4
                for i in range(4):
                    model.state_dict()[key][i * hidden_size:(i + 1) * hidden_size] = torch.from_numpy(
                        change_weights(
                            model.state_dict()[key].numpy()[i * hidden_size:(i + 1) * hidden_size], number))
        if torch.cuda.is_available():
            model.cuda()
        return model


def remove_node(model):
    # TODO: write code..
    return model


def mutate_model(model):
    mutated_model = LSTMDriver(model.input_size, model.hidden_size, model.num_layers, model.output_size,
                               model.batch_size)
    mutated_model.load_state_dict(model.state_dict())
    if torch.cuda.is_available():
        mutated_model.cuda()

    mutation_type = random.random()
    if mutation_type < 0.4:
        mutated_model = add_node_mutation(mutated_model)
    elif mutation_type >= 0.6:
        mutated_model = change_weights_mutation(mutated_model)
    else:
        mutated_model = remove_node(mutated_model)

    if torch.cuda.is_available():
        mutated_model.cuda()

    return mutated_model


def mutate_models(models):
    return [mutate_model(model) for model in models]


def multiply(model, times):
    models = []
    for _ in range(0, times):
        new_model = LSTMDriver(model.input_size, model.hidden_size, model.num_layers, model.output_size, model.batch_size)
        new_model.load_state_dict(model.state_dict())
        models.append(new_model)
    return models


def get_random_selection(models, sample_count=None):
    if sample_count is None:
        sample_count = len(models) // 2

    sample_models = []
    for i, model in enumerate(models):
        # Generate the reservoir
        if i < sample_count:
            sample_models.append(model)
        else:
            # Randomly replace elements in the reservoir
            # with a decreasing probability.
            # Choose an integer between 0 and i (inclusive)
            r = random.randint(0, i)
            if r < sample_count:
                sample_models[r] = model
    return sample_models


def get_best_models(models, sample_count=None):
    sample_count = len(models) // 2 if sample_count is None else sample_count
    model_score = score_models(models)
    return [model for model, score in sorted(model_score, key=lambda x: x[1], reverse=False)][:sample_count]


def get_initial_model():
    model = LSTMDriver(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE, BATCH_SIZE)
    model.load_state_dict(torch.load(INITIAL_LSTM_WEIGHTS_RELATIVE_PATH))
    if torch.cuda.is_available():
        model.cuda()
    return model


if __name__ == '__main__':
    lstm_models = mutate_models(multiply(get_initial_model(), POPULATION_SIZE))

    for index in range(0, NO_GENERATIONS):
        print('-------------------------------------------------------{} Generation---'.format(index+1))
        print("GET {} RANDOM MODELS".format(len(lstm_models) // 2))
        rand_sel_model = get_random_selection(lstm_models)
        print("GET BEST {} MODELS OUT OF {} RANDOM MODELS".format(len(rand_sel_model) // 2, len(rand_sel_model)))
        best_random_models = get_best_models(rand_sel_model)
        print("MUTATE BEST {} OF RANDOM MODELS".format(len(best_random_models)))
        mutated_best_random_models = mutate_models(best_random_models)
        print("TRAIN MUTATED BEST {} MODELS OUT OF {} RANDOM MODELS".format(len(mutated_best_random_models), len(rand_sel_model)))
        trnd_mut_best_rnd_models = train_models(mutated_best_random_models)
        print("PICK {} BEST MODELS OUT OF {}".format(POPULATION_SIZE, len(lstm_models) + len(trnd_mut_best_rnd_models)))
        lstm_models = get_best_models(lstm_models + trnd_mut_best_rnd_models, POPULATION_SIZE)

    best_model = get_best_models(lstm_models, 1)[0]

    best_model.cpu()
    torch.save(best_model.state_dict(), LSTM_EVOLUTION_WEIGHTS_PATH)
