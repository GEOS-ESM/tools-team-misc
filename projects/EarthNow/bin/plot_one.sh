#!/bin/bash

#NOTE: Running using the uv python config

# Add input arg to run various tests
# Check if arguments are provided
if [ "$#" -ne 3 ]; then
  echo "Error: Incorrect number of arguments."
  echo "Usage: $0 [conus|global] [YYYYMMDD] [product_name]"
  exit 1
fi

# Parse map type
INPUT="${1,,}"
if [[ "$INPUT" != "conus" && "$INPUT" != "global" ]]; then
  echo "Error: Invalid argument. Valid args: 'conus',  'global'."
  exit 1
fi

# Parse date
if [[ ! "$2" =~ ^[0-9]{8}$ ]]; then
  echo "Error: Invalid date format for the second argument. Please use YYYYMMDD (8 digits)."
  exit 1
fi

date="$2"
fdate=$date"_00z"
pdate=$date"_0000"

bindir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PRODUCT="$3"

# Generate plots from args
uv run "$bindir/plotall.py" \
  --product "$PRODUCT" \
  --nproc 1 \
  --fdate "$fdate" \
  --pdate "$pdate" \
  --map-type "$INPUT" \
  --base-path /discover/nobackup/"$USER"/EarthNow/plots \
  --style grey_topo
exit 0
