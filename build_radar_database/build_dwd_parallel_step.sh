#!/bin/bash

# This script should be called by build_dmi_parallel.sh with srun. It decompresses and processes one day of data.

concat_to_daily=$1
path=$2
location=$3
transfer=$4
jsc_folder=$5
overwrite=$6
file=$7

modes=("90gradstarng01" "pcpng01" "vol5minng01")
elevs=("00" "01" "02" "03" "04" "05" "06" "07" "08" "09")


if [ -f "$file" ]; then

    # extract year, month, day, and location from the file name
    filename1=$(basename "$file" _$location.tar)

    year=${filename1: -8:4}
    month=${filename1: -4:2}
    day=${filename1: -2:2}
    loc=${location}

    # create directory for this location if it doesn't exist
    midpath="$year/$year-$month/$year-$month-$day/$loc/"
    mkdir -p "$midpath"

    # check if the daily files ("*allmoms*") exist, if not, then extract
    extract=false


    for mode in ${modes[@]}; do


        for nn in ${elevs[@]}; do
            # for the scanning modes that are not the volume scan, we only have the 00 elevation so we skip the rest
            if [[ "$mode" = "90gradstarng01" || "$mode" = "pcpng01" ]] && [ "$nn" = "01" ]; then

                break

            fi

            if $transfer ; then #check if the daily files exist already, either locally or remotely according to transfer
                check_path="$jsc_folder$midpath/$mode/$nn/"
            else
                check_path="$midpath/$mode/$nn/"
            fi

            #if rsync $check_path/*allmoms* 1> /dev/null 2>&1; then
            if [ -e $check_path/*allmoms* ]; then
                # if file exist do nothing
                filesize=$(wc -c $check_path/*allmoms* | awk '{print $1}')

                # if file exists, check that it is larger than 1 mb, so we know it is not an erroneous file
                if [ $filesize -gt 1000000 ] && [ $overwrite = false ]; then
                    nothing=()
                else
                    extract=true
                    break 2
                fi
            else
                # if it does not exist, then break the loop and proceed to extract
                extract=true

                break 2

            fi


        done
    done




    # extract inner tar file and loop over all files
    if $extract; then
        # create temporary folder
        tempdir="$midpath"temp
        mkdir -p "$tempdir"

        # Unpack file
        echo "Unpacking ${file}"
        tar -xf "$file" -C "$midpath" --skip-old-files

        for inner_file in "$midpath"/*.tar.gz; do # loop through timestep-files

            # extract all inner files
            tar -xzf $inner_file -C "$tempdir" --strip-components=1 --skip-old-files

        done


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

        # sort files into corresponding folders
        for mode in ${modes[@]}; do

            for nn in ${elevs[@]}; do

                # for the scanning modes that are not the volume scan, we only have the 00 elevation so we skip the rest
                if [[ "$mode" = "90gradstarng01" || "$mode" = "pcpng01" ]] && [ "$nn" = "01" ]; then

                    break

                fi

                inner_path="$midpath/$mode/$nn/"
                mkdir -p $inner_path
                mv ${tempdir}/*$mode*_$nn* "$inner_path/."


            done
        done

        # delete inner file
        rm "$midpath"/*.tar.gz
        rm -r "$tempdir"


    fi


    ## PROCESSING TO DAILY FILES
    if $concat_to_daily && $extract ; then

        for mode in ${modes[@]}; do

            for nn in ${elevs[@]}; do

                # for the scanning modes that are not the volume scan, we only have the 00 elevation so we skip the rest
                if [[ "$mode" = "90gradstarng01" || "$mode" = "pcpng01" ]] && [ "$nn" = "01" ]; then

                    break

                fi

                inner_path="${midpath}${mode}/$nn/"

                ### check that the processed files do not exist already, if they are not found, proceed to process
                if ls $inner_path/*allmoms* 1> /dev/null 2>&1; then
                    # if file exist do nothing
                    filesize=$(wc -c $inner_path/*allmoms* | awk '{print $1}')

                    # if file exists, check if it is larger than 1 mb, so we know it is not an erroneous file
                    if [ $filesize -gt 1000000 ] && [ $overwrite = false ]; then
                        # if file exists and it is large, then we skip this case
                        continue
                    fi
                fi

                # check that there are files in the directory before processing
                if ls $inner_path/*hd5 1> /dev/null 2>&1; then
                    echo "   Processing *$mode*$nn*$year$month$day*$loc into daily file"

                    # Process timestep files into daily files per elevation with a python script (needs wradlib 1.19)
                    python concat_dwd_data_to_daily.py "$inner_path" && # run next command only if this finishes successfully

                    # remove timestep files
                    find . -type f -path "./$inner_path*hd5" ! -path "./$inner_path*allmoms*" -delete & # run in background
                fi


            done
        done

        # wait for all background commands to finish
        wait

        if $transfer; then
            # send to DETECT's storage in JSC and clean from local storage (run in the background):
            rsync -ar --progress --remove-source-files -R $midpath $jsc_folder >> "transfer_log_${loc}.txt" &
        fi

        wait
    fi

fi


