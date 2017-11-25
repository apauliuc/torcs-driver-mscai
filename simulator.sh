#!/bin/bash

counter=1

while [ $counter -le 100 ]
do
  torcs -r /home/andrei/Code/torcs-server/quickrace.xml &
  python torcs-client/run.py &
  wait
  ((counter++))
done

echo All done
