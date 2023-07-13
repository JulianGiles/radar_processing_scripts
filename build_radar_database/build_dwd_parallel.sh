#!/bin/bash

# This script runs several "steps" for computing several timesteps at the same time

# Same as build_database.sh but for selected years and months instead of the whole database

# Uncompresses all tar files from the given folder and sorts the files in DWD-like folder structure

# first I need to load a python environment with wradlib 1.19, numpy, datatree, sys, glob and tqdm
# run the following before executing this script:
# conda activate build_dwd_database

##############################################################
# Configurations for running in JUWELS:
##############################################################

#SBATCH --account=detectrea
#SBATCH --nodes=4
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=5
#SBATCH --time=14:00:00
#SBATCH --output=log2020b.out
#SBATCH --error=log2020b.err
#SBATCH --open-mode=append
#SBATCH --partition=mem192
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}

# which year(s) and month(s) to process
include_years=(2020)
include_months=(07 08 09 10 11 12)

concat_to_daily=true
path=/p/scratch/paj2301/dwd_raw/
location=pro # which location to extract
transfer=false # transfer to JSC? (deletes from local storage). If true, it will check if the files exist in JSC before doing anything, otherwise it will check locally
jsc_folder=/p/largedata2/detectdata/projects/A04/radar_data/dwd/
overwrite=false # what to do in case files are already found, if true then decompress and concat again and overwrite, if false skip this date


count=0

for file in ${path}${location}/*/*${location}.tar; do
    if [ -f "$file" ]; then
        # extract year, month, day, and location from the file name
        filename1=$(basename "$file" _$location.tar)

        year=${filename1: -8:4}
        month=${filename1: -4:2}
        day=${filename1: -2:2}
        loc=${location}

        # check if this is the year-month we want to process, otherwise continue
        if [[ ${include_years[@]} =~ $year ]] && [[ ${include_months[@]} =~ $month ]] ; then
            : # do nothing
        else
            continue
        fi

        ((count++))


        srun -c 8 --account=detectrea -n 1 --exact --threads-per-core=1 ./build_dwd_parallel_step.sh ${concat_to_daily} ${path} ${location} ${transfer} ${jsc_folder} ${overwrite} ${file} &

        # process only 20 days at the same time
        if [ "$count" -ge 20 ]; then
            wait

            count=0
        fi

    fi

done

wait


