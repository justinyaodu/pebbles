#!/bin/bash

FILES="./cvc4_test_inputs/*"
for file in $FILES
do
    echo "$file"
    start_time="$(date -u +%s.%N)"
    ./cvc4-1.7-win64-opt.exe $file
    end_time="$(date -u +%s.%N)"
    elapsed=$(echo "scale=3; $end_time - $start_time" | bc)
    echo -e "$elapsed seconds\n"
done