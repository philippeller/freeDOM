#!/bin/bash

N_X=150
SVC_LOG=logs/llh_service_log.txt

mkdir -p logs
rm -f "$SVC_LOG"

echo "# Starting llh_service..."
python llh_service.py >& $SVC_LOG &

printf "# Waiting until llh_service is initialized."
while ((1))
do
    grep "starting work loop" $SVC_LOG >/dev/null 2>&1 && break
    printf "."
    sleep 0.1
done
printf "\n"

start=`date +%s`

echo "# Starting clients..."
pids=()
i=0
for mu in -3 -2 -1 0 1 2 3
do
    for sig in 1 1.5 2 2.5 3 3.5 4
    do
        echo "# Launching client to test mu=${mu}, sig=${sig}..."
        python serial_opt_test.py $mu $sig $N_X >& logs/"serial_test_${i}_mu=${mu}_n_x=${N_X}_sig=${sig}.log" &
        pids+=($!)
        (( i++ ))
    done
done

echo "# Waiting for all clients to finish..."
for pid in ${pids[*]}
do
    wait $pid
done

end=`date +%s`
delta=`expr $end - $start`
echo "# total execution time was ${delta} seconds" 
echo "# Inspect plots to make sure output is reasonable."

echo "# Clients have finished. Telling llh service to stop."
python kill_service.py
