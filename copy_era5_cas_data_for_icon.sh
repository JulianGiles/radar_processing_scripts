#!/bin/bash

# Destination directory
DEST="/p/scratch/detectrea2/giles1/eur-0275_iconv2.6.4-eclm-parflowv3.12_wfe-case/."

# Your provided list of dates (raw input)
RAW_DATES="
2017-04-12
2017-06-30
2017-07-25
2017-10-05
2018-03-12
2018-04-16
2018-07-12
2019-10-04
2020-02-01*
2020-02-09*
2020-02-16*
2020-03-10*
2020-06-13*
2020-08-30
2020-09-25
2020-09-26*
2020-10-14
2020-10-30*
2016-02-21
2016-03-03
2016-11-30
2016-12-01
2016-12-26
2016-12-29
2016-12-30
2016-12-31
2017-03-03
2018-01-04
2019-12-24
2019-12-30
2020-01-03
2020-01-06
2020-01-07
2020-03-13 *
2020-03-17 *
2020-11-03 *
2020-11-04
2020-11-20
2020-12-14
2020-12-15 *
"

# Variable to store all calculated dates
ALL_DATES=""

# 1. Clean the input and calculate the previous day for each
while read -r line; do
    # Remove asterisks and spaces
    clean_date=$(echo "$line" | tr -d '* ' )

    # Skip empty lines
    if [ -z "$clean_date" ]; then continue; fi

    # Add the current date to our list
    ALL_DATES="$ALL_DATES$clean_date\n"

    # Calculate and add the previous date (GNU date format)
    prev_date=$(date -d "$clean_date - 1 day" +%Y-%m-%d)
    ALL_DATES="$ALL_DATES$prev_date\n"

done <<< "$RAW_DATES"

# 2. Sort the dates uniquely so we don't rsync the same day twice
UNIQUE_DATES=$(echo -e "$ALL_DATES" | sort -u | grep -v "^$")

# 3. Loop through the final unique list and run rsync
for target_date in $UNIQUE_DATES; do
    # Extract path and filename components
    YYYY=$(date -d "$target_date" +%Y)
    YYYY_MM=$(date -d "$target_date" +%Y_%m)
    YYYYMMDD=$(date -d "$target_date" +%Y%m%d)

    echo "=========================================="
    echo "Syncing data for: $target_date"

    # Construct source path and run rsync
    # We deliberately leave the wildcards unquoted so bash expands them
    rsync -avrum -R /p/data1/detectdata/CentralDB/./era5/${YYYY}/${YYYY_MM}/*${YYYYMMDD}* "$DEST"

done

echo "=========================================="
echo "All transfers completed successfully."
