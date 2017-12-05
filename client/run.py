#! /usr/bin/env python3

from pytocl.main import main
# from pytocl.driver import Driver
from lstm_andrei_driver import LSTMAndreiDriver

if __name__ == '__main__':
    main(LSTMAndreiDriver(logdata=False))
