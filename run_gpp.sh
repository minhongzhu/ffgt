#!/bin/bash

if [ $# -ne 3 ]; then
    echo "Usage: $0 <config> <seeds> <reps>"
    exit 1
fi

config=$1
seeds=$2
reps=$3

for ((i=seeds; i<seeds+reps; i++))
do
    echo "Running gpp.py with seed $i"
    python gpp.py --config "$config" --seed $i
done
