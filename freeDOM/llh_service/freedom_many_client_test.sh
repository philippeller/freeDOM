#!/bin/bash

N_CLIENTS=$1

SVC_LOG=logs/llh_service.log


echo "# N_CLIENTS=${N_CLIENTS}"

mkdir -p logs
rm -f "$SVC_LOG"


echo "# Starting llh_service..."
python ./llh_service.py > $SVC_LOG 2>&1 &
llh_svc_pid=$!

trap "kill $llh_svc_pid 2>/dev/null" 0


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
logfiles=()

for ((i = 0; i < $N_CLIENTS; i++))
do
    echo "# Launching client #${i}..."
    logfile=logs/"freedom_test_${i}.log"
    python ./freedom_test.py > "$logfile" 2>&1 &
    pids+=($!)
    logfiles[$i]="$logfile"
done


echo "# Waiting for all clients to finish..."
for pid in ${pids[*]}
do
    wait $pid
done


echo "# Clients have finished. Telling llh service to stop."
python ./kill_service.py


N_EVALS=0
T_MS=0
for ((i = 0; i < ${#logfiles[@]}; i++))
do
    logfile="${logfiles[$i]}"
    n_evals=$( cat "$logfile" | awk '{print $1}' )
    t_ms=$( cat "$logfile" | awk '{print $4}' )
    (( N_EVALS += n_evals ))
    T_MS=$( echo "$T_MS + $t_ms" | bc )
done


printf "# Average time per eval: %.0f us\n" $( echo "($T_MS * 1000) / ($N_EVALS * ${#logfiles[@]})" | bc )

end=`date +%s`
delta=`expr $end - $start`
echo "# Total execution time was ${delta} seconds"
#echo "# Inspect plots to make sure output is reasonable."
