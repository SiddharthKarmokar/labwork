#!/bin/bash

# Compile
echo "Compiling..."
gcc -O2 -fopenmp -mavx parallel_check_final.c -o parallel_check_final -lpthread

output_file="final_results.csv"
echo "Method,Size,Threads,Time" > $output_file

# Define parameters
sizes=(10000 50000 500000 2000000)
thread_counts=(2 4 8)

echo "Starting Experiments..."

for size in "${sizes[@]}"; do
    for threads in "${thread_counts[@]}"; do
        # Run Modes 1, 2, 3, 5 (Threaded modes)
        for mode in 1 2 3 5; do
            ./parallel_check_final $size $mode $threads >> $output_file
        done
        
        # Run Mode 4 (SIMD) only once per size (Threads don't affect it)
        if [ "$threads" -eq 2 ]; then
             ./parallel_check_final $size 4 1 >> $output_file
        fi
    done
done

echo "Experiments Complete. Plotting..."
python3 plot_final.py $output_file