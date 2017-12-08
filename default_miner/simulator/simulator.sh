#!/bin/bash

trial=(easy medium hard validation)
trial_count=0
while [ ${trial_count} -le 3 ]
do
    echo current trial: ${trial[${trial_count}]}
    race_count=0
    while [ ${race_count} -le 3 ]
    do
        python ./simulator/track_cfg.py ${trial[${trial_count}]} ${race_count}
        torcs -r /home/andrei/Code/torcs-driver-mscai/default_miner/simulator/quickrace.xml &
        python  run.py --difficulty ${trial[${trial_count}]}
        wait
        ((race_count++))
    done
    ((trial_count++))
done

echo All done