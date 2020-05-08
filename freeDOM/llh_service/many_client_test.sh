#!/bin/bash

N_PARALLEL=$1

N_X=22
SVC_LOG=logs/llh_service_log.txt

echo "# N_PARALLEL=${N_PARALLEL}"

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
logfiles=()
i=0
for mu in {-20..20}
do
    for sig in {1..20}
    do
        echo "# Launching client #${i} to test mu=${mu}, sig=${sig}..."
        logfile=logs/"serial_test_${i}_mu=${mu}_n_x=${N_X}_sig=${sig}.log"
        python serial_opt_test.py $mu $sig $N_X 2>&1 > "$logfile" &
        pids+=($!)
        logfiles[$i]="$logfile"
        (( i++ ))
        (( i >= N_PARALLEL )) && break
    done
    (( i >= N_PARALLEL )) && break
done

echo "# Waiting for all clients to finish..."
for pid in ${pids[*]}
do
    wait $pid
done

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
echo "# Inspect plots to make sure output is reasonable."

echo "# Clients have finished. Telling llh service to stop."
python kill_service.py
