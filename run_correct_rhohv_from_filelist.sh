#!/bin/bash
# This file runs the correct_rhohv.py script over several paths from a file

# Se the directory to look for the files
dir=/automount/realpep/upload/jgiles/dmi/

# Input text file containing file paths
input_file="${dir}to_recompute_rhonc.txt"

max_attempts=5  # Maximum number of restart attempts
max_execution_time=600  # Maximum execution time in seconds

# Loop through each file in the list
while IFS= read -r file; do
    attempt=1
    while [ $attempt -le $max_attempts ]; do

        # Invoke the Python script with the file path as an argument
        python /home/jgiles/Scripts/python/radar_processing_scripts/correct_rhohv.py $file &

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
        newfolder="${file/${dir}/${dir}rhohv_nc/}"
        newfolder=$(dirname "$newfolder")
        if [ -d "$newfolder" ]; then
            rm -r "$newfolder" # clean the created folder
        fi

        ((attempt++))
    done

    if [ $attempt -gt $max_attempts ]; then
        echo "Max restart attempts reached, could not be completed: $new_path"
    fi



done < "$input_file"

wait
