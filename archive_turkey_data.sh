#!/bin/bash

# This script archives the compressed data into year-location tarballs

# Base local data directory
DATA_DIR="/automount/realpep/upload/turkey-data"
REMOTE_USER_HOST="giles1@judac.fz-juelich.de"
REMOTE_BASE_DIR="/p/arch2/detectdata/a04/turkey-data"

cd "$DATA_DIR" || exit 1

# Loop over each year folder (2015 to 2020)
for year in 2015 2016 2017 2018 2019 2020; do
  if [ -d "$year" ]; then

    # Define remote directory path for the current year
    REMOTE_YEAR_DIR="${REMOTE_BASE_DIR}/${year}"

    # Create the remote year directory if it doesn't exist yet
    ssh "$REMOTE_USER_HOST" "mkdir -p '$REMOTE_YEAR_DIR'"

    # Change into the year directory locally so paths inside tar start at LOCATION/
    (
      cd "$year" || exit 1

      # Loop over location folders (e.g., ANK)
      for loc_path in */; do
        [ -d "$loc_path" ] || continue

        # Remove trailing slash to get clean name (e.g., "ANK")
        loc_name="${loc_path%/}"
        archive_name="${loc_name}.tar"

        echo "=== Archiving $year/$loc_name -> $REMOTE_YEAR_DIR/$archive_name ==="

        # Archive local folder 'ANK' and stream directly to remote year path
        tar -cf - "$loc_name" | ssh "$REMOTE_USER_HOST" "cat > '$REMOTE_YEAR_DIR/$archive_name'"

        if [ $? -eq 0 ]; then
          echo "Successfully archived: $year/$loc_name"
        else
          echo "ERROR: Failed streaming $year/$loc_name" >&2
        fi

      done
    )

  fi
done

echo "=== Archiving complete. Target remote structure recreated. ==="
