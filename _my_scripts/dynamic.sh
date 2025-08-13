#!/bin/bash

# Check if start and end parameters are provided
if [ $# -ne 2 ]; then
    echo "Usage: $0 <start> <end>"
    echo "Example: $0 0 14"
    exit 1
fi

start=$1
end=$2

# Validate that start and end are numbers
if ! [[ "$start" =~ ^[0-9]+$ ]] || ! [[ "$end" =~ ^[0-9]+$ ]]; then
    echo "Error: start and end must be positive integers"
    exit 1
fi

# Validate that start <= end
if [ $start -gt $end ]; then
    echo "Error: start ($start) must be less than or equal to end ($end)"
    exit 1
fi

echo "Processing chunks from $start to $end"

# Activate the conda environment
conda activate qhflow
# cd ~/25DFT/QHFlow/src

start_time=$(date +%s)

for i in $(seq $start $end); do
    chunk_start=$(date +%s)
    echo "Starting chunk $i at $(date)"
    
    python -m dataset_module.qh9_datasets_split --name=QH9Dynamic --num_chunks=30 --chunk_idx=$i --split=mol
    
    if [ $? -eq 0 ]; then
        chunk_end=$(date +%s)
        chunk_duration=$((chunk_end - chunk_start))
        echo "Successfully completed chunk $i in ${chunk_duration} seconds"
    else
        echo "Failed on chunk $i"
        exit 1
    fi
done

end_time=$(date +%s)
total_duration=$((end_time - start_time))
echo "Total execution time: ${total_duration} seconds"

