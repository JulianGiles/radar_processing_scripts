#!/bin/bash

# Uncompresses all tar files from the folder /p/largedata2/detectdata/projects/A04/2022-Nov-new/ and sorts the files in DWD-like folder structure

# first I need to load a python environment with wradlib 1.19, numpy, datatree, sys, glob and tqdm
# run the following before executing this script:
# conda activate build_dwd_database

concat_to_daily=true
path=/automount/realpep/upload/RealPEP-SPP/newdata/2022-Nov-new/
location=pro # which location to extract
transfer=true # transfer to JSC? (deletes from local storage). If true, it will check if the files exist in JSC before doing anything, otherwise it will check locally
jsc_folder=giles1@judac.fz-juelich.de:/p/largedata2/detectdata/projects/A04/radar_data/dwd/

modes=("90gradstarng01" "pcpng01" "vol5minng01")
elevs=("00" "01" "02" "03" "04" "05" "06" "07" "08" "09")


for file in ${path}${location}/*${location}.tar; do
    if [ -f "$file" ]; then

        # extract year, month, day, and location from the file name
        filename1=$(basename "$file" .tar)

        year=${filename1:0:4}
        month=${filename1:4:2}
        day=${filename1:6:2}
        loc=${filename1:9:3}

        # create directory for this location if it doesn't exist
        midpath="$year/$year-$month/$year-$month-$day/$loc/"
        mkdir -p "$midpath"

        # check if the daily files ("*allmoms*") exist, if not, then extract
        extract=false

        for mode in ${modes[@]}; do

            if [ "$mode" = "90gradstarng01" ]; then
                # we skip this mode because it is not always present
                continue
            fi

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

                if rsync $check_path/*allmoms* 1> /dev/null 2>&1; then
                    # if file exist do nothing
                    nothing=()
                else
                    # if it does not exist, then break the loop and proceed to extract

                    extract=true

                    break 2

                fi


            done
        done


        # extract inner tar file and loop over all files
        if $extract; then
            # Unpack file
            echo "Unpacking ${file}"
            tar -xf "$file" -C "$midpath" --skip-old-files

            # create temporary folder
            mkdir "$midpath"temp

            for inner_file in "$midpath"/*.tar.gz; do # loop through timestep-files

                # extract all inner files
                tar -xzf $inner_file -C "$midpath"temp --strip-components=1

            done

            # sort files into corresponding folders
            for mode in ${modes[@]}; do

                for nn in ${elevs[@]}; do

                    # for the scanning modes that are not the volume scan, we only have the 00 elevation so we skip the rest
                    if [[ "$mode" = "90gradstarng01" || "$mode" = "pcpng01" ]] && [ "$nn" = "01" ]; then

                        break

                    fi

                    inner_path="$midpath/$mode/$nn/"
                    mkdir -p $inner_path
                    mv ${midpath}temp/*$mode*_$nn* "$inner_path/."


                done
            done

            # delete inner file
            rm "$midpath"/*.tar.gz
            rm -r "$midpath"temp


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
                        nothing=()
                    else

                        # check that there are files in the directory before processing
                        if ls $inner_path/*hd5 1> /dev/null 2>&1; then
                            echo "   Processing *$mode*$nn*$year$month$day*$loc into daily file"

                            # Process timestep files into daily files per elevation with a python script (needs wradlib 1.19)
                            time python concat_dwd_data_to_daily.py "$inner_path" && # run next command only if this finishes successfully

                            # remove timestep files
                            find . -type f -path "./$inner_path*hd5" ! -path "./$inner_path*allmoms*" -delete & # run in background
                        fi
                    fi

                done
            done

            # wait for all background commands to finish
            wait

            if $transfer; then
                # send to DETECT's storage in JSC and clean from local storage (run in the background):
                rsync -ar --progress --remove-source-files -R $midpath $jsc_folder >> "transfer_log_${loc}.txt" &
            fi

        fi
    fi

done


