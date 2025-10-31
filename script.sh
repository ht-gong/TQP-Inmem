#!/bin/bash

# Loop through values of i from 1 to 22
for i in {1..22}
do
    # Run the Python script and save the output to a file named output_i.txt
    # numactl --cpunodebind=0 --membind=0 python3 main.py --q $i --SF=100 --h 30 --log experiments/results/2025-05-15/output_${i}.json
    python3 experiments/parse_log.py experiments/results/2025-05-15/output_${i}.json /work1/talati/haotiang/TQP-Vortex/experiments/figures Q${i}
done
