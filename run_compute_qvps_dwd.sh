#!/bin/bash

# This script runs compute_qvps_new.py over files matching the pattern

# first I need to load a python environment with wradlib, numpy, datatree, sys, glob and tqdm

# Set the directory to look for the files
dir=/automount/realpep/upload/jgiles/dwd/

# Which location to process
loc=tur

max_attempts=5  # Maximum number of restart attempts
max_execution_time=240  # Maximum execution time in seconds

# Create a list of all files that include *allmoms* in their name
files=$(find $dir -name "*allmoms_07*$loc*" -type f -not -path "*qvp*"  -not -path "*WIND*" -not -path "*SURVEILLANCE*" -not -path "*RHI1*" | sort -u)

# Loop through each file in the list
for file in $files; do

    attempt=1
    while [ $attempt -le $max_attempts ]; do

        # Invoke the Python script with the file path as an argument
        python /home/jgiles/Scripts/python/radar_processing_scripts/compute_qvps_new.py $file &

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
        newfolder="${file/${dir}/${dir}qvps/}"
        newfolder=$(dirname "$newfolder")
        if [ -d "$newfolder" ]; then
            rm -r "$newfolder" # clean the created folder
        fi

        ((attempt++))
    done

    if [ $attempt -gt $max_attempts ]; then
        echo "Max restart attempts reached, could not be completed: $file"
    fi

done

wait
