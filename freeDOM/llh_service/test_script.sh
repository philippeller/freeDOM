#!/bin/bash


N_X=27
SVC_LOG=logs/llh_service_log.txt

mkdir -p logs
rm -f "$SVC_LOG"

echo "# Starting llh_service..."
python llh_service.py >& $SVC_LOG &
llh_svc_pid=$!

trap "kill $llh_svc_pid" 0


printf "# Waiting until llh_service is initialized."
while ((1))
do
    grep "starting work loop" $SVC_LOG >/dev/null 2>&1 && break
    printf "."
    sleep 0.1
done
printf "\n"


echo "# Starting clients..."
pids=()
i=0
for mu in 0.5 -0.5 0.25 -0.25 0 0.1 -0.1
do
    echo "# Launching client to test mu=${mu}..."
    python client_test.py $mu $N_X &  #>& logs/"client_test_${i}_mu=${mu}_n_x=${N_X}.log" &
    pids+=($!)
    (( i++ ))
done


echo "# Waiting for all clients to finish..."
for pid in ${pids[*]}
do
    wait $pid
done


echo "# Inspect plots to make sure output is reasonable."
echo "# Clients have finished. Telling llh service to stop."
python kill_service.py
