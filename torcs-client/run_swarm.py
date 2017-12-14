#! /usr/bin/env python3
from pytocl.main import main
from my_driver_swarm import MyDriver

import sys
if __name__ == '__main__':   
    main(MyDriver(logdata=False, port=sys.argv[2]))



