#!/bin/bash
# This file runs the correct_rhohv.py script over several paths

# Se the directory to look for the files
dir=/automount/realpep/upload/jgiles/dwd/

# Create a list of all files that include *allmoms* in their name
files=$(find $dir -name "*allmoms*20150101*pro*" -type f)

# Loop through each file in the list
for file in $files; do

  # Pass the file path to the python script
  python /home/jgiles/Scripts/python/radar_processing_scripts/correct_rhohv.py $file

done
