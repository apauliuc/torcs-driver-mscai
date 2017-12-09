#! /usr/bin/env python3

from pytocl.main import main
# from drivers.my_driver import MyDriver
# from drivers.esn_driver import ESNDriver
from drivers.lstm_driver import Driver_LSTM

if __name__ == '__main__':
    main(Driver_LSTM(logdata=False))
