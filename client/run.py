#! /usr/bin/env python3

from pytocl.main import main
# from client.pytocl.driver import Driver
from my_driver_esn import MyDriver

if __name__ == '__main__':
    main(MyDriver(logdata=False))
