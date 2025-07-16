#!/bin/bash

# Define the base directory and the output directory
BASE_DIR="/automount/realpep/upload/jgiles/ICON_EMVORADO_radardata/eur-0275_iconv2.6.4-eclm-parflowv3.12_wfe-case/"
dates=("2016-02-21"
"2016-03-03"
"2016-11-30"
"2016-12-01"
"2016-12-26"
"2016-12-29"
"2016-12-30"
"2016-12-31"
"2017-03-03"
"2018-01-04"
"2019-12-24"
"2019-12-30"
"2020-01-03"
"2020-01-06"
"2020-01-07"
"2020-11-04"
"2020-11-20"
"2020-12-14") # hty dates
dates=("2017-04-12" "2017-06-30" "2017-07-25" "2017-10-05" "2018-03-12" "2018-04-16" "2018-07-12" "2019-10-04" "2020-08-30" "2020-09-25" "2020-10-14") # pro dates
loc="pro"

for date in "${dates[@]}"; do
    find "$BASE_DIR" -type d -path "*$date/$loc/vol" ! -path "*/qvps/*" | while read -r path; do
        echo "launching: compute_qvps_new_emvorado.py $path/"
        python -u /home/jgiles/Scripts/python/radar_processing_scripts/compute_qvps_new_emvorado.py "$path/"
    done
done
