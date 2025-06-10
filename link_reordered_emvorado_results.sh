#!/bin/bash

# Define the base directory and the output directory
BASE_DIR="/automount/realpep/upload/jgiles/ICON_EMVORADO_test/eur-0275_iconv2.6.4-eclm-parflowv3.12_wfe-case/run/"
FIND_PATTERN="*icon_??????????/*/cdfin_allsim_*"
OUTPUT_DIR="/automount/realpep/upload/jgiles/ICON_EMVORADO_offline_radardata/eur-0275_iconv2.6.4-eclm-parflowv3.12_wfe-case/"
LOC_CODES=("010392:pro" "010832:tur" "010356:umd" "017187:afy" "017138:ank" "017259:gzt" "017373:hty" "017163:svs")
RUN_NAME="icon" # icon or iconemvorado

# Overwrite flag
OVERWRITE=${1:-no}

# Create an associative array for location codes
declare -A locs_code
for loc_code in "${LOC_CODES[@]}"; do
  key="${loc_code%%:*}"
  value="${loc_code##*:}"
  locs_code[$key]=$value
done

# Find all matching files
find "$BASE_DIR" -type f -path "$FIND_PATTERN" | while read -r file; do
  # Extract forecast start time, location ID, and timestamp from the file path
  #forecast_time=$(echo "$file" | grep -oP "${RUN_NAME}_\K\d{10}")
  #timestep=$(basename "$file" | grep -oP "_\K\d{12}(?=_\d{12})")
  timestep=$(basename "$file" | grep -oP "_\K\d{12}" | head -n 1)
  location_id=$(basename "$file" | grep -oP "id-\K\d+")

  # Convert the timestep to a formatted date
  YYYY=$(echo "$timestep" | cut -c 1-4)
  MM=$(echo "$timestep" | cut -c 5-6)
  DD=$(echo "$timestep" | cut -c 7-8)

  # Get the location code
  loc=${locs_code[$location_id]}

  # Skip if the location is not in the dictionary
  if [ -z "$loc" ]; then
    echo "Skipping unknown location ID: $location_id"
    continue
  fi

  # Construct the output directory structure
  target_dir="$OUTPUT_DIR/$YYYY/$YYYY-$MM/$YYYY-$MM-$DD/$loc/vol"
  mkdir -p "$target_dir"

  # Determine the target symbolic link
  target_link="$target_dir/$(basename "$file")"

  # Handle overwrite logic
  if [ -L "$target_link" ]; then
    if [ "$OVERWRITE" == "yes" ]; then
      echo "Overwriting existing link: $target_link"
      ln -sf "$file" "$target_link"
    else
      echo "Skipping existing link: $target_link"
    fi
  else
    ln -s "$file" "$target_link"
  fi
done
