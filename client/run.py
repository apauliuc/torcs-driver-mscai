#! /usr/bin/env python3

from pytocl.main import main
<<<<<<< HEAD
# from pytocl.driver import Driver
from lstm_andrei_driver import LSTMAndreiDriver

if __name__ == '__main__':
    main(LSTMAndreiDriver(logdata=False))
=======
# from client.pytocl.driver import Driver

# from my_driver_esn import MyDriver
from my_driver_lstm import MyDriver

if __name__ == '__main__':
    main(MyDriver(logdata=False))
>>>>>>> 28be1112c985094b9189cb6a0d79e147072ce818
