#!/bin/bash
# This file runs the calibrate_zdr.py script over several paths

# Set the directory to look for the files
dir="/automount/realpep/upload/jgiles/dwd/20*"

# Set the type of calibration method
calibtype=1

# Create a list of all files that include *allmoms* in their name
files=$(find $dir -name "*90gradstarng01*allmoms*pro*" -type f)

# Loop through each file in the list
for file in $files; do

  # Pass the file path to the python script
  python /home/jgiles/Scripts/python/radar_processing_scripts/calibrate_zdr.py $file $calibtype

done
