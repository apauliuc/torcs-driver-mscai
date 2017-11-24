#! /usr/bin/env python3

from pytocl.main import main
from my_driver import MyDriver
from pytocl.driver import Driver

if __name__ == '__main__':
    main(Driver(logdata=False))
