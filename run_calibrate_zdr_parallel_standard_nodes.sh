#!/bin/bash

# This script runs several "steps" of calibrate_zdr.py script over several paths

# first I need to load a python environment with wradlib 1.19, numpy, datatree, sys, glob and tqdm

##############################################################
# Configurations for running in JUWELS:
##############################################################

#SBATCH --account=detectrea
#SBATCH --nodes=4
#SBATCH --cpus-per-task=6
#SBATCH --ntasks-per-node=8
#SBATCH --time=20:00:00
#SBATCH --job-name=svs_calib_zdr_lr
#SBATCH --output=calibrate_zdr_lr_svs.out
#SBATCH --error=calibrate_zdr_lr_svs.err
#SBATCH --open-mode=append
#SBATCH --partition=batch
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}

# Set the directory to look for the files
dir=/p/scratch/detectrea/giles1/radar_data/dmi/

# Set the type of calibration method
calibtype=2

# set a name for the counter file (counting how many job steps running at the same time
counterfile=$dir/count_svs.txt

# Create a list of all files that include *allmoms* in their name
files=$(find $dir -name "*allmoms*SVS*" -type f -not -path "*qvp*" -not -path "*WIND*" -not -path "*SURVEILLANCE*" -not -path "*RHI1*")

count=0
echo $count > $counterfile
startcount=0
# Loop through each file in the list
for file in $files; do

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
        { timeout 10m srun -c 6 --account=detectrea -n 1 --exact --threads-per-core=1  python $dir/calibrate_zdr.py $file $calibtype; count=$(<$counterfile); ((count--)) ; echo $count > $counterfile; } &

        if [ "$startcount" -le 30 ]; then
            sleep 5
        fi
    fi

    while [ "$(<$counterfile)" -ge 30 ]; do
        sleep 5
    done

done

wait
