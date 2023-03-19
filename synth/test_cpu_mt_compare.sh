#!/bin/bash

FILES="./random_5s_10exs/*"
TIMEFORMAT=%R


for (( OMP_NUM_THREADS = 1; OMP_NUM_THREADS <= 16; OMP_NUM_THREADS++ ));
do
    echo "$OMP_NUM_THREADS threads"
    for file in $FILES
    do
        echo "$file"
        time_taken=$((time timeout 300s ./synth_cpu_mt_input_file $file) 2>&1 >/dev/null)
        echo "$time_taken" >>/dev/stderr
    done
done
