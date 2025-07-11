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
#SBATCH --job-name=generate_volumes_hty_detecticonemvorado
#SBATCH --output=generate_volumes_hty_detecticonemvorado.out
#SBATCH --error=generate_volumes_hty_detecticonemvorado.err
#SBATCH --open-mode=truncate
#SBATCH --partition=batch
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}

# Set the directory with the code (this will be the working dir)
codedir=/p/scratch/detectrea/giles1/radar_processing_scripts/
cd $codedir

# Define the radar code
radar_code="017373" # pro: "010392" hty: "017373"

# Define your list of dates (format: YYYYMMDDHH)
dates=(
    "2019122400"
    "2019123000"
    "2020010300"
    "2020010600"
    "2020010700"
    "2020110400"
    "2020112000"
    "2020121400"
)
# Set the directory to look for the files
base_path="/p/scratch/detecticonemvorado/eur-0275_iconv2.6.4-eclm-parflowv3.12_wfe-case/run/"

# set a name for the counter file (counting how many job steps running at the same time
counterfile=$base_path/count_hty.txt

count=0
echo $count > $counterfile
startcount=0
# Loop through each date
for date in "${dates[@]}"; do

    # Build paths
    path_radout="${base_path}/iconemvorado_${date}/radout/"
    path_radout_filtered="${base_path}/iconemvorado_${date}/radout_filtered/"
    path_icon="${base_path}/iconemvorado_${date}/"
    path_icon_z="${base_path}/iconemvorado_${date}/"
    path_save="${base_path}/iconemvorado_${date}/icon_vol/"

    # Link files from radout into radout_filtered only if they correspond to the date after spin-up time (otherwise there is no ICON data to match them)
    mkdir -p $path_radout_filtered
    ln -sf $path_radout/*$radar_code*${date:0:8}* $path_radout_filtered/.

    count=$(<$counterfile)
    ((count++))
    echo $count > $counterfile
    ((startcount++))
    # Pass the file path to the python script
    { srun -c 4 --account=detectrea -n 1 --exact --threads-per-core=1 --time 1200 python -u $codedir/generate_icon_volumes.py "$radar_code" "$path_radout_filtered" "$path_icon" "$path_icon_z" "$path_save"; count=$(<$counterfile); ((count--)) ; echo $count > $counterfile; } &

    if [ "$startcount" -le 12 ]; then
        sleep 5
    fi

    while [ "$(<$counterfile)" -ge 12 ]; do
        sleep 5
    done

done

wait
