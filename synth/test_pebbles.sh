#!/bin/bash

FILES="./random_5s_10exs/*"
for file in $FILES
do
    echo "$file"
    time ./synth_cpu_mt_input_file $file
done
