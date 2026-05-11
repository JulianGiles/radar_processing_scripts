#!/bin/bash

# This script runs several "steps" of calibrate_zdr.py script over several paths

# first I need to load a python environment with wradlib 1.19, numpy, datatree, sys, glob and tqdm

##############################################################
# Configurations for running in JUWELS:
##############################################################

#SBATCH --account=detectrea2
#SBATCH --nodes=8
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=6
#SBATCH --time=03:00:00
#SBATCH --job-name=compute_qvps
#SBATCH --output=compute_qvps.out
#SBATCH --error=compute_qvps.err
#SBATCH --open-mode=truncate
#SBATCH --partition=batch
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}

# Set the directory with the code (this will be the working dir)
codedir=/p/scratch/detectrea2/giles1/radar_processing_scripts/
cd $codedir

# Set the directory to look for the files
dir=/p/scratch/detectrea2/giles1/dwd/

# set a name for the counter file (counting how many job steps running at the same time
counterfile=$dir/count_qvps.txt

# Create a list of all files that include *allmoms* in their name
files=$(find $dir -name "*vol5minng01*allmoms_07*umd*" -type f -not -path "*qvp*"  -not -path "*ppi*" | sort -u) # "*vol5minng01*allmoms_07*umd*"

count=0
echo $count > $counterfile
startcount=0
# Loop through each file in the list
for file in $files; do

    # Get the last folder of the file path (indicating elevation) # REMEMBER DWD ELEVATION NUMBER IS NOT THE ACTUAL ELEVATION!
    elev=$(basename "$(dirname "$file")")

    # Function to check if the number is between 7 and 15
    is_between_0_and_15() {
    local number_float=$1

    # Use bc to convert the number string to float
    local float_value=$(echo "scale=2; $number_float" | bc)

    # Check if the float value is between 7 and 15 (inclusive)
    (( $(bc <<< "$float_value >= 0.0 && $float_value <= 50.0") ))
    }

    # Check if the elevation is a valid float and within the desired range
    if [[ $elev =~ ^[0-9]+(\.[0-9]+)?$ ]] && is_between_0_and_15 "$elev"; then
        count=$(<$counterfile)
        ((count++))
        echo $count > $counterfile
        ((startcount++))
        # Pass the file path to the python script
        { timeout 30m srun -c 8 --account=detectrea2 -n 1 --exact --threads-per-core=1 python -u $codedir/compute_qvps_new.py $file; count=$(<$counterfile); ((count--)) ; echo $count > $counterfile; } &

        if [ "$startcount" -le 48 ]; then
            sleep 5
        fi
    fi

    while [ "$(<$counterfile)" -ge 48 ]; do
        sleep 5
    done

done

wait
