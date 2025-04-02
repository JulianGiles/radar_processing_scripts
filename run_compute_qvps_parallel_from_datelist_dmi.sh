#!/bin/bash

# This script runs several "steps" of calibrate_zdr.py script over several paths

# first I need to load a python environment with wradlib 1.19, numpy, datatree, sys, glob and tqdm

##############################################################
# Configurations for running in JUWELS:
##############################################################

#SBATCH --account=detectrea
#SBATCH --nodes=4
#SBATCH --cpus-per-task=12
#SBATCH --ntasks-per-node=4
#SBATCH --time=12:00:00
#SBATCH --job-name=compute_qvps_svs
#SBATCH --output=compute_qvps_svs.out
#SBATCH --error=compute_qvps_svs.err
#SBATCH --open-mode=truncate
#SBATCH --partition=batch
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}

# Set the directory to look for the files
dir=/p/scratch/detectrea/giles1/radar_data/dmi/

# set a name for the counter file (counting how many job steps running at the same time
counterfile=$dir/count_svs.txt

# Create a list of all files that include *allmoms* in their name
loc=SVS
files=$(find $dir -name "*allm*SVS*" -type f -not -path "*qvp*"  -not -path "*WIND*" -not -path "*SURVEILLANCE*" -not -path "*RHI1*" -not -path "*ppi*" | sort -u)

date_file="/p/scratch/detectrea/giles1/radar_data/dmi/svs_selected_dates.txt"

# Read the dates from the file into an array
mapfile -t valid_dates < "$date_file"

count=0
echo $count > $counterfile
startcount=0
# Loop through each file in the list
for file in $files; do

    # Extract date using $loc as a reference
    path_before_loc=${file%/$loc*}  # Get everything before loc
    file_date=$(basename "$path_before_loc")  # Get the last part, which is the date

    # Check if the extracted date is in the list
    if [[ " ${valid_dates[*]} " =~ " $file_date " ]]; then

        # Get the last folder of the file path (indicating elevation)
        elev=$(basename "$(dirname "$file")")

        # Function to check if the number is between 10 and 15
        is_between_7_and_15() {
        local number_float=$1

        # Use bc to convert the number string to float
        local float_value=$(echo "scale=2; $number_float" | bc)

        # Check if the float value is between 10 and 15 (inclusive)
        (( $(bc <<< "$float_value >= 7.0 && $float_value <= 15.0") ))
        }

        # Check if the elevation is a valid float and within the desired range
        if [[ $elev =~ ^[0-9]+(\.[0-9]+)?$ ]] && is_between_7_and_15 "$elev"; then
            count=$(<$counterfile)
            ((count++))
            echo $count > $counterfile
            ((startcount++))
            # Pass the file path to the python script
            { srun -c 12 --account=detectrea -n 1 --exact --threads-per-core=1 --time 10 python $dir/compute_qvps_new.py $file; count=$(<$counterfile); ((count--)) ; echo $count > $counterfile; } &

            if [ "$startcount" -le 30 ]; then
                sleep 5
            fi
        fi

        while [ "$(<$counterfile)" -ge 30 ]; do
            sleep 5
        done
    fi
done

wait
