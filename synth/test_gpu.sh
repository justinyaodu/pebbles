#!/bin/bash

FILES="./random_5s/*"
TIMEFORMAT=%R

echo
for file in $FILES
do
    echo "$file"
    time_taken=$((time timeout 300s ./synth_gpu_input_file $file) 2>&1 >/dev/null)
    echo "$time_taken" >>/dev/stderr
done
