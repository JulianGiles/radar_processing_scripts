#!/bin/bash

# This script should be called by build_dmi_parallel.sh with srun. It decompresses and processes one day of data.

include_years=$1
include_months=$2
concat_to_daily=$3
path=$4
location=$5
transfer=$6
jsc_folder=$7
overwrite=$8
file=$9

if [ -f "$file" ]; then

    # extract year, month, day, and location from the file name
    filename1=$(basename "$file" .tar.gz)

    year=${filename1: -8:4}
    month=${filename1: -4:2}
    day=${filename1: -2:2}
    loc=${location}

    # WE HAD TO TAKE THIS PART OUT BECAUSE YOU CANNOT PASS ARRAYS FROM ONE SCRIPT TO ANOTHER
    # check if this is the year-month we want to process, otherwise continue
    # if [[ ${include_years[@]} =~ $year ]] && [[ ${include_months[@]} =~ $month ]] ; then
    #     : # do nothing
    # else
    #     exit
    # fi


    # create directory for this location if it doesn't exist
    midpath="$year/$year-$month/$year-$month-$day/$loc/"
    mkdir -p "$midpath"

    # check if the daily files ("*allmoms*") exist, if not, then extract
    extract=false


    if $transfer ; then #check if the daily files exist already, either locally or remotely according to transfer
        check_path="$jsc_folder$midpath"
    else
        check_path="$midpath"
    fi

    if rsync -q --list-only $check_path/*/*/*allmoms* 1> /dev/null 2>&1 && [ $overwrite = "false" ] ; then
        # if file exist and overwrite==false do nothing
        nothing=()
    else
        # if it does not exist or overwrite is true, then proceed to extract

        extract=true

    fi




    # extract inner tar file and loop over all files
    if $extract; then
        # create temporary folder
        tempdir="$midpath"temp
        mkdir -p "$tempdir"

        # Unpack file
        echo "Unpacking ${file}"
        tar -xf "$file" -C "$midpath"temp --skip-old-files

        # in case the files are inside a folder structure, pull them up to the temp folder
        # (originally I used the --strip-components in the tar command but not always there is a tree folder structure inside
        if [ -d "$midpath"temp/acq ]; then
            # The folder exists, do something
            mv "$midpath"temp/acq/*/*/*/*/*/*/*/*/* "$midpath"temp
        fi

        # check that something was actually extracted and that the tar file was not empty
        if [ "$(ls -A $tempdir)" ]; then
            # if not empty, do nothing
            nothing=()
        else
            # if empty, report and continue
            echo "tar file is empty"
            rm -r "$tempdir"
            exit

        fi

    fi


    ## PROCESSING TO DAILY FILES
    if $concat_to_daily && $extract ; then

        # check that there are files in the directory before processing
        if ls $tempdir/*RAW* 1> /dev/null 2>&1; then
            echo "   Processing $year$month$day*$loc into daily files"

            # Process timestep files into daily files per elevation with a python script (needs wradlib 1.19)
            python concat_dmi_data_to_daily.py "$tempdir" "${midpath}/concat/" && # run next command only if this finishes successfully

            # remove timestep files
            find . -type f -path "./$tempdir*RAW" ! -path "./$tempdir*allmoms*" -delete & # run in background
        fi

        # wait for all background commands to finish
        wait

        # move the final files and delete temporary folders
        if [ "$(ls -A ${midpath}/concat/)" ]; then
            # first, check that files were actually created

            for concatfile in "${midpath}/concat/"*allmoms*; do
                # Extract the values of mode and elev from the filename
                mode=$(basename "$concatfile" | cut -d'-' -f1)
                elev=$(basename "$concatfile" | cut -d'-' -f3)

                # Create the directories for mode and elev, if they don't exist
                mkdir -p "$midpath/$mode/$elev"

                # Move the file to the appropriate directory
                mv "$concatfile" "$midpath/$mode/$elev/"
            done
        fi

        rm -r "$tempdir"
        rm -r "${midpath}/concat/"

        if $transfer; then
            # send to DETECT's storage in JSC and clean from local storage (run in the background):
            rsync -ar --progress --remove-source-files -R $midpath $jsc_folder >> "transfer_log_${loc}.txt" &
        fi

    fi
fi


