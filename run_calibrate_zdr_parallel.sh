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
#SBATCH --time=24:00:00
#SBATCH --output=calibrate_zdr_pro.out
#SBATCH --error=calibrate_zdr_pro.err
#SBATCH --open-mode=append
#SBATCH --partition=mem192
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}

# Set the directory to look for the files
dir=/p/scratch/detectrea/giles1/radar_data/dwd/

# set a name for the counter file (counting how many job steps running at the same time
counterfile=$dir/count_pro.txt

# Set the type of calibration method
calibtype=2

# Create a list of all files that include *allmoms* in their name
files=$(find $dir -name "*vol5minng01*allmoms*pro*" -type f -not -path "*qvp*")

count=0
echo $count > $counterfile
startcount=0
# Loop through each file in the list
for file in $files; do

    count=$(<$counterfile)
    ((count++))
    echo $count > $counterfile
    ((startcount++))
    # Pass the file path to the python script
    { srun -c 6 --account=detectrea -n 1 --exact --threads-per-core=1  python $dir/calibrate_zdr.py $file $calibtype; count=$(<$counterfile); ((count--)) ; echo $count > $counterfile; } &

    if [ "$startcount" -le 30 ]; then
        sleep 5
    fi

    while [ "$(<$counterfile)" -ge 30 ]; do
        sleep 5
    done

done

wait
