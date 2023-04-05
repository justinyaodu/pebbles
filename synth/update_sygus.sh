#!/bin/bash

FILES="./cvc4_test_inputs/*"
TIMEFORMAT=%R

echo
for file in $FILES
do
	FILENAME=$(basename "$file")
	echo "$FILENAME"
    ./sygus-v1-to-v2.sh ./cvc4-1.8-x86_64-linux-opt "$file" "./outputs/$FILENAME"
done
