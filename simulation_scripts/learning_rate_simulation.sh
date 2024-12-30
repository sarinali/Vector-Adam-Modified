#!/bin/bash

learning_rates=(0.5)

start=0.6
end=1.5
num_steps=10

current_date=$(date +"%Y%m%d_%H%M%S")

# populate learning rates
current=$start
while (( $(echo "$current <= $end" | bc -l) )); do
    learning_rates+=($current)
    current=$(echo "$current + 0.1" | bc)
done

# run the script for all learning rates
for rate in "${learning_rates[@]}"; do
    echo "Current Learning Rate: $rate"
    /Users/sarinali/.local/share/virtualenvs/VectorAdam-rkohprO_/bin/python /Users/sarinali/Projects/VectorAdam/demo2_transport_vs_project.py 1 $num_steps --learning_rate $rate >> logs/${current_date}_loss_logs.txt 2>&1
done