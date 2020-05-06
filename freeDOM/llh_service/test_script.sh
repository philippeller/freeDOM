#!/bin/bash

mkdir -p logs

python llh_service.py >& logs/llh_service_log.txt &
python client_test.py 0.5 1000 >& logs/pt5_test.txt &
pid1=$!
python client_test.py -0.5 800 >& logs/negpt5_test.txt &
pid2=$!
python client_test.py 0.25 300 >& logs/pt25_test.txt &
pid3=$!
python client_test.py -0.25 100 >& logs/negpt25_test.txt 
pid4=$!

wait $pid1
wait $pid2
wait $pid3
wait $pid4

python kill_service.py
