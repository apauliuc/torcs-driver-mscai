#! /usr/bin/env python3

from pytocl.main import main
from drivers.lstm_driver import LSTMDriver

if __name__ == '__main__':
    main(LSTMDriver(logdata=False))
