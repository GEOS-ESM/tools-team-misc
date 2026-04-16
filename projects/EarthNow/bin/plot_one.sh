#!/bin/bash

#NOTE: Running using the uv python config

# Add input arg to run various tests
# Check if arguments are provided
# if [ "$#" -ne 4 ]; then
#   echo "Error: Incorrect number of arguments."
#   echo "Usage: $0 [conus|global] [YYYYMMDD] [product_name] [single | all]"
#   exit 1
# fi

if [ "$#" -ne 2 ]; then
  echo "Error: Incorrect number of arguments."
  echo "Usage: $0 [conus|global] [single | all]"
  exit 1
fi

# Parse map type
MAP_TYPE="${1,,}"
if [[ $MAP_TYPE != "conus" && $MAP_TYPE != "global" ]]; then
  echo "Error: Invalid map type argument. Valid args: 'conus',  'global'."
  exit 1
fi

# Parse date
# if [[ ! $2 =~ ^[0-9]{8}$ ]]; then
#   echo "Error: Invalid date format (YYYYMMD)"
#   exit 1
# fi

FDATE="20260408"
FDATE="$FDATE""_00z"

PRODUCT="vorticity_heights_500mb_EarthNow"

FRAMES="${2,,}"
if [[ $FRAMES != "single" && $FRAMES != "all" ]]; then
  echo "Error: Invalid frame argument. Valid args: 'single',  'all' (frames)."
  exit 1
fi

bindir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "$FRAMES" = "single" ]]; then
  # Generate single plot
  uv run "$bindir/plotall.py" \
    --product "$PRODUCT" \
    --nproc 1 \
    --fdate "$FDATE" \
    --pdate "$FDATE" \
    --map-type "$MAP_TYPE" \
    --base-path /discover/nobackup/"$USER"/EarthNow/plots \
    --style grey_topo
  exit 0
else
  # Generate all plots
  uv run "$bindir/plotall.py" \
    --product "$PRODUCT" \
    --nproc 48 \
    --fdate "$FDATE" \
    --map-type "$MAP_TYPE" \
    --base-path /discover/nobackup/"$USER"/EarthNow/plots \
    --style grey_topo
  exit
fi
