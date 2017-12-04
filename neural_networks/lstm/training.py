from neural_networks.lstm.LSTM import LSTM
from neural_networks.lstm.data_processing import load_training_data
import glob
import os
import numpy as np
import pickle

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


dtype = torch.cuda.FloatTensor
continue_train = False

if __name__ == '__main__':
    script_dir = os.path.dirname(__file__)
    project_dir = os.path.split(os.path.split(script_dir)[0])[0]

    # driver = 'default_driver'
    # train_data_path = os.path.join(project_dir, '/data/csv/{}/*.csv'.format(driver))
    train_data_path = os.path.join(project_dir, 'data/csv')

    training_files = glob.glob(train_data_path + '/*.csv')

    training_sets = load_training_data(training_files)

    params = {
        'input': 22,
        'hidden': 50,
        'output': 3,
        'layers': 3
    }

    save_obj(params, 'rnn_params')

    INPUT_SIZE = params['input']
    HIDDEN_SIZE = params['hidden']
    OUTPUT_SIZE = params['output']
    NUM_LAYERS = params['layers']
    BATCH_SIZE = 1
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.00001

    lstm_nn = LSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE, BATCH_SIZE)

    if continue_train:
        lstm_nn.load_state_dict(torch.load('weight_params.pt'))

    lstm_nn.cuda()

    # Same learning rate
    criterion = nn.MSELoss().cuda()
    optimizer = torch.optim.Adam(lstm_nn.parameters(), lr=LEARNING_RATE)

    for epoch in np.arange(NUM_EPOCHS):
        epoch_error = 0
        print('Epoch [%d/%d]' % (epoch + 1, NUM_EPOCHS))
        for f in training_files:
            train_loader = DataLoader(dataset=training_sets[f], batch_size=BATCH_SIZE, shuffle=False)
            lstm_nn.init_hidden()

            for i, (X, y) in enumerate(train_loader):
                if len(X) != BATCH_SIZE:
                    continue

                data = Variable(X.view(-1, 1, INPUT_SIZE)).type(dtype)
                target = Variable(y).type(dtype)

                optimizer.zero_grad()
                prediction = lstm_nn(data)
                loss = criterion(prediction, target)
                loss.backward()
                optimizer.step()

                epoch_error += loss.data[0]

                # if (i + 1) % BATCH_SIZE == 0:
                #     print('    step: [%d/%d], loss: %.4f'
                #           % (i + 1, len(training_sets[f].target_tensor) // BATCH_SIZE, loss.data[0]))

        print('Epoch error: {}\n'.format(epoch_error))

    print('Training done')

    lstm_nn.cpu()
    torch.save(lstm_nn.state_dict(), 'weight_params.pt')
