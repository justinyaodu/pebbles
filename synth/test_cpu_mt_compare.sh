#!/bin/bash

FILES="./random_5s_10exs/*"
TIMEFORMAT=%R


for (( NUM_THREADS = 1; NUM_THREADS <= 8; NUM_THREADS++ ));
do
    echo "$NUM_THREADS threads"
    for file in $FILES
    do
        echo "$file"
        time_taken=$((OMP_NUM_THREADS=$NUM_THREADS time timeout 300s ./synth_cpu_mt_input_file $file) 2>&1 >/dev/null)
        echo "$time_taken" >>/dev/stderr
    done
done
