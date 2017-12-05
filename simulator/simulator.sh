#!/bin/bash

counter=1

while [ ${counter} -le 9 ]
do
  python ./simulator/track_cfg.py ${counter}
  torcs -r /home/andrei/Code/torcs-driver-mscai/simulator/quickrace.xml &
  python  ./client/run.py
#  python2 ./snakeoil_2015/client.py --stage 0 --track ${counter} --steps 100000 --port 3001 --host localhost

#  torcs -r /home/andrei/Code/torcs-driver-mscai/simulator/quickrace.xml &
#  python2 ./snakeoil_2015/client.py --stage 2 --track ${counter} --steps 100000 --port 3001 --host localhost &
  wait
  ((counter++))
done

echo All done
