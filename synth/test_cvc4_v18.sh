#!/bin/bash

FILES="./cvc4_test_inputs/*"
TIMEFORMAT=%R

echo
for file in $FILES
do
    echo "$file"
    time_taken=$((time timeout 300s ./cvc4-1.8-x86_64-linux-opt --lang=sygus1 $file) 2>&1 >/dev/null)
    echo -n "$time_taken " >>/dev/stderr
    time_taken=$((time timeout 300s ./synth_cpu_st_input_file $file) 2>&1 >/dev/null)
    echo "$time_taken" >>/dev/stderr
done
