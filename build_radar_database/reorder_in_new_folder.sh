#!/bin/bash

# Set the path to the directory containing the original folders
src_dir=/automount/realpep/upload/RealPEP-SPP/newdata/2022-Nov-new/

# Set the path to the directory where you want to create the new folder structure
dest_dir=giles1@judac.fz-juelich.de:/p/largedata2/detectdata/projects/A04/radar_data/dwd_raw/

sites=(tur)

# Loop through the sites
for rs in ${sites[@]}; do
    # Loop through each folder in the source directory
    for folder in $src_dir/$rs/*; do
        # Check if the item in the loop is actually a directory
        if [[ -d $folder ]]; then
            # Loop through each tar file in the folder
            for file in $folder/*.tar; do
                # Check if the item in the loop is actually a file
                if [[ -f $file ]]; then
                    # Get the location code from the tar file name
                    loctar=$(basename $file | cut -d '_' -f 2)
                    loc=${loctar:0:3}
                    # Get the date from the tar file name
                    date=$(basename $file | cut -d '_' -f 1)
                    year=${date:0:4}
                    month=${date:4:2}
                    day=${date:6:2}
                    # Copy the file with rsync
                    rsync -avr --progress $file $dest_dir/$loc/$year-$month-$day/
                fi
            done
        fi
    done
done
