#!/bin/bash

# This script runs several compute_qvps_new.py over a list of paths

# first I need to load a python environment with wradlib 1.19, numpy, datatree, sys, glob and tqdm

# Set the directory to look for the files
dir=/automount/realpep/upload/jgiles/dwd/

# Input text file containing file paths
input_file="${dir}folders_ML_reprocess_umdfail.txt"

# Which location to process
loc=umd

# Loop through each file path in the input file
while IFS= read -r file_path; do
    # if it is a qvp path, remove the "qvps" part
    if [[ $file_path == qvps/* ]]; then
        new_path="$dir${file_path#qvps/}"
    else
        new_path="$dir$file_path"
    fi

    if [[ $file_path != */$loc/* ]]; then
        continue
    fi

    # Check if the file path is not empty
    if [ -n "$new_path" ]; then
        # Invoke the Python script with the file path as an argument
        python /home/jgiles/Scripts/python/radar_processing_scripts/compute_qvps_new.py "$new_path"
    fi
done < "$input_file"

wait
