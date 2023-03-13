#!/bin/bash

FILES="./cvc4_test_inputs/*"
for file in $FILES
do
    echo "$file"
    time timeout 300s ./cvc4-1.7-x86_64-linux-opt $file --lang=sygus1 $file
done