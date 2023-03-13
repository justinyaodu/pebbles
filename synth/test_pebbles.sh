#!/bin/bash

FILES="./cvc4_test_inputs/*"
for file in $FILES
do
    echo "$file"
    time ./synth_cpu_st_input_file.exe $file
done