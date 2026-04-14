#!/bin/bash

#NOTE: Running using the uv python config

# Add input arg to run various tests
# Check if exactly one argument is provided
if [ -z "$1" ]; then
  echo "Error: No argument provided."
  echo "Usage: $0 [conus|global]"
  exit 1
fi

# Convert the input to lowercase (to handle 'CONUS', 'Global', etc.)
INPUT="${1,,}"
if [[ "$INPUT" != "conus" && "$INPUT" != "global" ]]; then
  echo "Error: Invalid argument. Valid args: 'conus',  'global'."
  exit 1
fi

bindir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Set fdate/pdate variables
# date="20260202"
date="20260324"
fdate=$date"_00z"
pdate=$date"_0000"

# Set product variable
PRODUCT="vorticity_heights_500mb_EarthNow"

# Generate plots from args
uv run "$bindir/plotall.py" \
  --product "$PRODUCT" \
  --nproc 1 \
  --fdate $fdate \
  --pdate $pdate \
  --map-type "$INPUT" \
  --base-path /discover/nobackup/"$USER"/EarthNow/plots \
  --style grey_topo
exit 0
