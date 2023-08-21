#!/bin/bash

# This script runs several compute_qvps_new.py over a list of paths

# first I need to load a python environment with wradlib 1.19, numpy, datatree, sys, glob and tqdm

# Set the directory to look for the files
dir=/automount/realpep/upload/jgiles/dmi/

# Input text file containing file paths
input_file="${dir}folders_SVS.txt"

# Which location to process
loc=SVS

max_attempts=5  # Maximum number of restart attempts
max_execution_time=120  # Maximum execution time in seconds

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
        attempt=1
        while [ $attempt -le $max_attempts ]; do

            # Invoke the Python script with the file path as an argument
            python /home/jgiles/Scripts/python/radar_processing_scripts/compute_qvps_new.py "$new_path" &

            # Get the process ID of the background script
            script_pid=$!

            # set time counter and steps
            timecount=0
            sleepterval=5

            while [ $timecount -le $max_execution_time ]; do
                # sleep for a bit
                sleep $sleepterval
                timecount=$(( $timecount + $sleepterval ))

                # Check if the script is still running
                if ps -p $script_pid > /dev/null; then
                    : # do nothing
                else
                    # Script finished successfully, break out of the loops
                    break 2
                fi

            done

            # if after the max time the script is still running, kill it
            kill $script_pid
            echo "Script exceeded time limit. Killing and retrying..."
            # clean the created folder
            if [ -d "$dir$file_path" ]; then
                rm -r "$dir/qvps/$file_path" # clean the created folder
            fi

            ((attempt++))
        done

        if [ $attempt -gt $max_attempts ]; then
            echo "Max restart attempts reached, could not be completed: $new_path"
        fi

    fi

done < "$input_file"

wait
