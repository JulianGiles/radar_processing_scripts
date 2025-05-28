#!/bin/bash

# This script runs several "steps" of generate_icon_volumes.py script over several paths

# first I need to load a python environment with wradlib 1.19, numpy, datatree, sys, glob and tqdm

##############################################################
# Configurations for running in JUWELS:
##############################################################

#SBATCH --account=detectrea
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=12
#SBATCH --time=24:00:00
#SBATCH --job-name=generate_volumes_pro
#SBATCH --output=generate_volumes_pro.out
#SBATCH --error=generate_volumes_pro.err
#SBATCH --open-mode=truncate
#SBATCH --partition=mem192
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}

# Set the directory with the code (this will be the working dir)
codedir=/p/scratch/detectrea/giles1/radar_processing_scripts/
cd $codedir

# Define the radar code
radar_code="010392"

# Define your list of dates (format: YYYYMMDDHH)
dates=(
    "2017041200"
    "2017063000"
    "2017072500"
    "2017100500"
    "2018031200"
    "2018041600"
    "2018071200"
    "2019100400"
    "2020083000"
    "2020092500"
    "2020101400"
)
# Set the directory to look for the files
base_path="/p/scratch/detectrea/giles1/eur-0275_iconv2.6.4-eclm-parflowv3.12_wfe-case/run/"

# set a name for the counter file (counting how many job steps running at the same time
counterfile=$base_path/count_pro.txt

count=0
echo $count > $counterfile
startcount=0
# Loop through each date
for date in "${dates[@]}"; do

    # Build paths
    path_radar="${base_path}/icon_${date}/radarout/"
    path_icon="${base_path}/icon_${date}/"
    path_icon_z="${base_path}/icon_${date}/"
    path_save="${base_path}/icon_${date}/icon_vol/"

    count=$(<$counterfile)
    ((count++))
    echo $count > $counterfile
    ((startcount++))
    # Pass the file path to the python script
    { srun -c 4 --account=detectrea -n 1 --exact --threads-per-core=1 --time 600 python -u $codedir/generate_icon_volumes.py "$radar_code" "$path_radar" "$path_icon" "$path_icon_z" "$path_save"; count=$(<$counterfile); ((count--)) ; echo $count > $counterfile; } &

    if [ "$startcount" -le 12 ]; then
        sleep 5
    fi

    while [ "$(<$counterfile)" -ge 12 ]; do
        sleep 5
    done

done

wait
