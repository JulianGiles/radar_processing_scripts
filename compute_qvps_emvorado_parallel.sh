#!/bin/bash

# This script runs several "steps" of compute_qvps_new_emvorado.py script over several paths. First the files need to be arranged in a DWD-like folder structure.

# first I need to load a python environment with wradlib 1.19, numpy, datatree, sys, glob and tqdm

##############################################################
# Configurations for running in JUWELS:
##############################################################

#SBATCH --account=detectrea
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=12
#SBATCH --time=24:00:00
#SBATCH --job-name=compute_qvps_hty
#SBATCH --output=compute_qvps_hty.out
#SBATCH --error=compute_qvps_hty.err
#SBATCH --open-mode=truncate
#SBATCH --partition=mem192
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}

# Set the directory with the code (this will be the working dir)
codedir=/p/scratch/detectrea/giles1/radar_processing_scripts/
cd $codedir

# Define the radar site
loc="hty"

# Define your list of dates (format: YYYY-MM-DD)
dates=(
    "2016-02-21"
    "2016-03-03"
    "2016-11-30"
    "2016-12-26"
    "2016-12-29"
    "2016-12-30"
    "2016-12-31"
    "2017-03-03"
    "2018-01-04"
    "2019-12-24"
    "2019-12-30"
    "2020-01-03"
    #"2020-01-06"
    "2020-11-04"
    "2020-11-20"
    "2020-12-14"
)
# Set the directory to look for the files
base_path="/p/scratch/detecticonemvorado/eur-0275_iconv2.6.4-eclm-parflowv3.12_wfe-case/radar_data_ordered/"

# set a name for the counter file (counting how many job steps running at the same time
counterfile=$base_path/count_hty.txt

count=0
echo $count > $counterfile
startcount=0
# Loop through each date
for date in "${dates[@]}"; do

    find "$base_path" -type d -path "*$date/$loc/vol" ! -path "*/qvps/*" | while read -r path; do
        echo "launching: compute_qvps_new_emvorado.py $path/"

        count=$(<$counterfile)
        ((count++))
        echo $count > $counterfile
        ((startcount++))
        # Pass the file path to the python script
        { srun -c 4 --account=detectrea -n 1 --exact --threads-per-core=1 --time 20 python -u $codedir/compute_qvps_new_emvorado.py "$path/"; count=$(<$counterfile); ((count--)) ; echo $count > $counterfile; } &

        if [ "$startcount" -le 12 ]; then
            sleep 5
        fi

        while [ "$(<$counterfile)" -ge 12 ]; do
            sleep 5
        done
    done

done

wait
