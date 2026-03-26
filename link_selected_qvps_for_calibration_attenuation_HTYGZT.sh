#!/bin/bash

# This script takes a list of dates and links the corresponding qvp files to a parallel folder, so they are easier to select in other scripts.

# Base directories
SOURCE_BASE="/automount/realpep/upload/jgiles/dmi/qvps"
TARGET_BASE="/automount/realpep/upload/jgiles/dmi/qvps_selected_for_calibration_attenuation"

# List of dates to process (in YYYY-MM-DD format)
DATES=(
    "2016-02-06"
    "2016-12-14"
    "2016-12-16"
    "2016-12-20"
    "2016-12-21"
    "2016-12-22"
    "2016-12-27"
    "2016-12-29"
    "2016-12-30"
    "2016-12-31"
    "2017-01-01"
    "2017-01-02"
    "2020-01-19"

    # New dates
    "2016-12-25"
    "2016-12-26"
    "2017-01-03"
    "2020-01-16"
    "2020-01-17"
    "2020-01-20"
    "2017-12-24"
    "2019-12-28"
    "2019-12-31"
    "2020-01-02"
    "2020-01-03"
    "2020-01-07"
    "2020-02-07"
    "2020-02-29"
    "2020-03-18"
    "2020-03-19"
    "2020-03-20"
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
