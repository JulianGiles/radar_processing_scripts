#!/bin/bash

# This script takes a list of dates and links the corresponding qvp files to a parallel folder, so they are easier to select in other scripts.

# Base directories
SOURCE_BASE="/automount/realpep/upload/jgiles/dmi/qvps"
TARGET_BASE="/automount/realpep/upload/jgiles/dmi/qvps_selected_for_emvorado"

# List of dates to process (in YYYY-MM-DD format)
DATES=("2017-04-12" "2017-06-30" "2017-07-25" "2017-10-05" "2018-03-12" "2018-04-16" "2018-07-12" "2019-10-04" "2020-08-30" "2020-09-25" "2020-10-14") #PRO
DATES=(
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
) #HTY

# Loop through each date
for date in "${DATES[@]}"; do
    # Extract year and year-month from the date
    year="${date:0:4}"
    year_month="${date:0:7}"

    # Source directory path
    src_dir="${SOURCE_BASE}/${year}/${year_month}/${date}"

    if [ -d "$src_dir" ]; then
        # Find and link all files under this date's directory
        find "$src_dir" -type f | while read -r file; do
            # Compute relative path and target path
            relative_path="${file#$SOURCE_BASE/}"
            target_path="${TARGET_BASE}/${relative_path}"

            # Create target directory if it doesn't exist
            mkdir -p "$(dirname "$target_path")"

            # Link the file if it doesn't already exist
            if [ ! -e "$target_path" ]; then
                ln -s "$file" "$target_path"
                echo "Linked: $file -> $target_path"
            else
                echo "Already exists: $target_path"
            fi
        done
    else
        echo "Source directory does not exist: $src_dir"
    fi
done
