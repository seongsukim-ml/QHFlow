#!/bin/bash

# Activate the conda environment
conda activate qhflow
cd ~/25DFT/QHFlow/src

start_time=$(date +%s)

for i in {1..14}; do
    chunk_start=$(date +%s)
    echo "Starting chunk $i at $(date)"
    
    python -m dataset_module.qh9_datasets_split --name=QH9Stable --num_chunks=30 --chunk_idx=$i
    
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

