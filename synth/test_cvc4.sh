#!/bin/bash

FILES="./cvc4_test_inputs/*"
for file in $FILES
do
    echo "$file"
    #start_time="$(date -u +%s.%N)"
    time timeout 300s ./cvc4-1.7-x86_64-linux-opt $file
    #end_time="$(date -u +%s.%N)"
    #elapsed=$(echo "scale=3; $end_time - $start_time" | bc)
    #echo -e "$elapsed seconds\n"
done
