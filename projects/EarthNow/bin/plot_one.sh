#!/bin/bash

#NOTE: Running using the uv python config

# Add input arg to run various tests
# Check if exactly one argument is provided
if [ -z "$1" ]; then
    echo "Error: No argument provided."
    echo "Usage: $0 {conus|global}"
    exit 1
fi

# Convert the input to lowercase (to handle 'CONUS', 'Global', etc.)
INPUT="${1,,}"

bindir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Test uv setup works
# script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# project_root="$(dirname "$script_dir")"
# cd "$project_root"
# uv run tests/test_uv.py
# 

# Test GLOBAL
if [ "$INPUT" == "global" ]; then
  uv run "$bindir/plotall.py" \
      --product vorticity_heights_500mb_EarthNow \
      --nproc 1 \
      --fdate 20260202_00z \
      --pdate 20260202_1600 \
      --map-type global \
      --base-path /discover/nobackup/$USER/EarthNow/plots \
      --style grey_topo
  exit 0

# Test CONUS
elif [ "$INPUT" == "conus" ]; then
  uv run "$bindir/plotall.py" \
      --product vorticity_heights_500mb_EarthNow \
      --nproc 1 \
      --fdate 20260202_00z \
      --pdate 20260202_1600 \
      --map-type conus \
      --base-path /discover/nobackup/$USER/EarthNow/plots \
      --style grey_topo
  exit 0

else
  # Catch-all for invalid arguments
  echo "Error: Invalid argument '$1'."
  echo "Usage: $0 {conus|global}"
  exit 1

fi
