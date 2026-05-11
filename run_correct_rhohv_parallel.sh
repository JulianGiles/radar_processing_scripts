#!/bin/bash

# This script runs several "steps" of correct_rhohv.py script over several paths

# first I need to load a python environment with wradlib 1.19, numpy, datatree, sys, glob and tqdm

##############################################################
# Configurations for running in JUWELS:
##############################################################

#SBATCH --account=detectrea2
#SBATCH --nodes=8
#SBATCH --cpus-per-task=6
#SBATCH --ntasks-per-node=8
#SBATCH --time=06:00:00
#SBATCH --output=correct_rhohv.out
#SBATCH --error=correct_rhohv.err
#SBATCH --open-mode=truncate
#SBATCH --partition=batch
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}

# Set the directory with the code (this will be the working dir)
codedir=/p/scratch/detectrea2/giles1/radar_processing_scripts/
cd $codedir

# Set the directory to look for the files
dir=/p/scratch/detectrea2/giles1/dmi/

# set a name for the counter file (counting how many job steps running at the same time
counterfile=$dir/count_rhohv.txt

# Create a list of all files that include *allmoms* in their name
files=$(find $dir -name "*allmom*" -type f -not -path "*qvp*"  -not -path "*WIND*" -not -path "*RHI1*" -not -path "*ppi*" | sort -u)


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
    { srun -c 6 --account=detectrea2 -n 1 --exact --threads-per-core=1 --time 30  python -u $codedir/correct_rhohv.py $file; count=$(<$counterfile); ((count--)) ; echo $count > $counterfile; } &

    if [ "$startcount" -le 30 ]; then
        sleep 5
    fi

    while [ "$(<$counterfile)" -ge 30 ]; do
        sleep 5
    done

done

wait
