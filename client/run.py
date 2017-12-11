#! /usr/bin/env python3

from pytocl.main import main
from pytocl.driver import Driver
# from drivers.my_driver import MyDriver
# from drivers.esn_driver import ESNDriver
from drivers.lstm_driver import LSTMDriver

if __name__ == '__main__':
    main(LSTMDriver(logdata=False))
