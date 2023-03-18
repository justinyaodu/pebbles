#!/bin/bash

FILES="./cvc4_test_inputs/*"
for file in $FILES
do
    echo "$file"
    ./synth_cpu_st_input_file $file
done
