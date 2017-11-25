#!/bin/bash

counter=1

while [ $counter -le 1 ]
do
  torcs -r /home/andrei/Code/torcs-server/simulator/quickrace.xml &
  python ./track_cfg.py ${counter}
  python ../snakeoil2015/client.py &
  wait
  ((counter++))
done

echo All done
