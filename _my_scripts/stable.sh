#!/bin/bash

# Check if start and end parameters are provided (optional third arg: root)
if [ $# -lt 2 ] || [ $# -gt 3 ]; then
    echo "Usage: $0 <start> <end> [root]"
    echo "Example: $0 0 14 /home/gimseongsu/shared/QHFlow/dataset"
fi

start=$1
end=$2

# Root directory (optional arg 3, defaults to existing path)
if [ $# -eq 3 ]; then
    root="$3"
else
    root="/home/holywater2/25DFT/QHFlow"
fi

# Validate that start and end are numbers
if ! [[ "$start" =~ ^[0-9]+$ ]] || ! [[ "$end" =~ ^[0-9]+$ ]]; then
    echo "Error: start and end must be positive integers"
fi

# Validate that start <= end
if [ $start -gt $end ]; then
    echo "Error: start ($start) must be less than or equal to end ($end)"
fi

echo "Processing chunks from $start to $end"
echo "Using root: $root"

# Validate root directory exists
if [ ! -d "$root" ]; then
    echo "Error: root directory does not exist: $root"
fi

# Activate the conda environment
conda activate qhflow
cd $root/src

start_time=$(date +%s)

for i in $(seq $start $end); do
    chunk_start=$(date +%s)
    echo "Starting chunk $i at $(date)"
    
    python -m dataset_module.qh9_datasets_split --name=QH9Stable --num_chunks=30 --chunk_idx=$i
        
    if [ $? -eq 0 ]; then
        chunk_end=$(date +%s)
        chunk_duration=$((chunk_end - chunk_start))
        echo "Successfully completed chunk $i in ${chunk_duration} seconds"
    else
        echo "Failed on chunk $i"
    fi
done

end_time=$(date +%s)
total_duration=$((end_time - start_time))
echo "Total execution time: ${total_duration} seconds"

