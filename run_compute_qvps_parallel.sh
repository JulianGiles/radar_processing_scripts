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
#SBATCH --time=01:00:00
#SBATCH --job-name=compute_qvps_umd
#SBATCH --output=compute_qvps_umd.out
#SBATCH --error=compute_qvps_umd.err
#SBATCH --open-mode=truncate
#SBATCH --partition=batch
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}

# Set the directory with the code (this will be the working dir)
codedir=/p/scratch/detectrea/giles1/radar_processing_scripts/
cd $codedir

# Set the directory to look for the files
dir=/p/scratch/detectrea/giles1/dwd/

# set a name for the counter file (counting how many job steps running at the same time
counterfile=$dir/count_umd.txt

# Create a list of all files that include *allmoms* in their name
files=$(find $dir -name "*vol5minng01*allmoms_07*umd*" -type f -not -path "*qvp*"  -not -path "*ppi*" | sort -u)

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
    { srun -c 12 --account=detectrea -n 1 --exact --threads-per-core=1 --time 10 python $codedir/compute_qvps_new.py $file; count=$(<$counterfile); ((count--)) ; echo $count > $counterfile; } &

    if [ "$startcount" -le 30 ]; then
        sleep 5
    fi

    while [ "$(<$counterfile)" -ge 30 ]; do
        sleep 5
    done

done

wait
