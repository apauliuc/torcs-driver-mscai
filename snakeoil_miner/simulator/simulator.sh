#!/bin/bash

trial=(easy medium hard validation)
trial_count=0
while [ ${trial_count} -le 3 ]
do
    echo current trial: ${trial[${trial_count}]}
    race_count=0
    while [ ${race_count} -le 3 ]
    do
        python2 ./simulator/track_cfg.py ${trial[${trial_count}]} ${race_count}
        torcs -r /home/andrei/Code/torcs-driver-mscai/snakeoil_2015/simulator/quickrace.xml &
        python2 client.py --stage 0 --track ${trial_count}${race_count} --steps 100000 --port 3001 --host localhost --difficulty ${trial[${trial_count}]}

        torcs -r /home/andrei/Code/torcs-driver-mscai/snakeoil_2015/simulator/quickrace.xml &
        python2 client.py --stage 2 --track ${trial_count}${race_count} --steps 100000 --port 3001 --host localhost --difficulty ${trial[${trial_count}]} &
        wait
        ((race_count++))
    done
    ((trial_count++))
done

echo All done
