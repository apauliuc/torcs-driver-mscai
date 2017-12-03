from neural_networks.lstm.LSTM import LSTM
from neural_networks.lstm.data_processing import load_training_data
import glob
import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable

continue_train = True

if __name__ == '__main__':
    script_dir = os.path.dirname(__file__)
    project_dir = os.path.split(os.path.split(script_dir)[0])[0]

    # driver = 'default_driver'
    # train_data_path = os.path.join(project_dir, '/data/csv/{}/*.csv'.format(driver))
    train_data_path = os.path.join(project_dir, 'data/csv')

    training_files = glob.glob(train_data_path + '/bot*.csv')

    training_sets = load_training_data(training_files)

    INPUT_SIZE = 28
    HIDDEN_SIZE = 56
    NUM_LAYERS = 3
    BATCH_SIZE = 100
    NUM_EPOCHS = 10
    LEARNING_RATE = 0.00001

    lstm_nn = LSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, 3, BATCH_SIZE)

    if continue_train:
        lstm_nn.load_state_dict(torch.load('rnn_params.pt'))

    # Same learning rate
    criterion = nn.MSELoss()
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

                data = Variable(X.view(-1, 1, INPUT_SIZE))
                target = Variable(y)

                optimizer.zero_grad()
                prediction = lstm_nn(data)
                loss = criterion(prediction, target)
                loss.backward()
                optimizer.step()

                epoch_error += loss.data[0]

                if (i + 1) % BATCH_SIZE == 0:
                    print('    step: [%d/%d], loss: %.4f'
                          % (i + 1, len(training_sets[f].target_tensor) // BATCH_SIZE, loss.data[0]))

        print('Epoch error: {}\n'.format(epoch_error))

    print('Training done')

    torch.save(lstm_nn.state_dict(), 'rnn_params.pt')
