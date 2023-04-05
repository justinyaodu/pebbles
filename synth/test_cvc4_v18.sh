#!/bin/bash

FILES="./cvc5_inputs/*"
TIMEFORMAT=%R

echo
for file in $FILES
do
    echo "$file"
    time_taken=$((time timeout 300s ./cvc5-Linux --lang=sygus $file) 2>&1 >/dev/null)
    #echo -n "$time_taken " >>/dev/stderr
    #time_taken=$((time OMP_NUM_THREADS=4 timeout 300s ./synth_cpu_mt_input_file $file) 2>&1 >/dev/null)
    echo "$time_taken" >>/dev/stderr
done
