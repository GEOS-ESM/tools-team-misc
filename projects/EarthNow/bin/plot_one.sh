#!/bin/bash

#NOTE: Running using the uv python config

# Add input arg to run various tests
# Check if arguments are provided
if [ "$#" -ne 6 ]; then
  echo "Error: Incorrect number of arguments."
  echo "Usage: ${BASH_SOURCE[0]} <conus|global> <Style> <forecast time as YYYYMMDD_HHz> <frame time as YYYYMMDD_HHHHz> <product_name> <single|all>"
  exit 1
fi

# Parse map type
MAP_TYPE="${1,,}"
if [[ $MAP_TYPE != "conus" && $MAP_TYPE != "global" ]]; then
  echo "Error: Invalid map type argument. Valid args: 'conus',  'global'."
  exit 1
fi

# Parse map type
STYLE_TYPE=$2

# Parse date
# FDATE defines the *directory* (forecast initialization time) 
# and PDATE defines the frame/timestamp to plot
FDATE="${3,,}"
if [[ ! $FDATE =~ ^[0-9]{8}_[0-9]{2}z$ ]]; then
  echo "Error: Invalid first date format (YYYYMMDD_HHz)"
  exit 1
fi

PDATE="${4,,}"
if [[ ! $PDATE =~ ^[0-9]{8}_[0-9]{4}z$ ]]; then
  echo "Error: Invalid second date format (YYYYMMDD_HHHHz)"
  exit 1
fi

PRODUCT=$5

FRAMES="${6,,}"
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
    --pdate "$PDATE" \
    --map-type "$MAP_TYPE" \
    --base-path /discover/nobackup/"$USER"/EarthNow/plots \
    --style "$STYLE_TYPE"
  exit 0
else
  # Generate all plots
  if  [[ -n "$SLURM_CPUS_PER_TASK" ]]; then
    nproc="$SLURM_CPUS_PER_TASK"
  else
    nproc=1
  fi
  uv run "$bindir/plotall.py" \
    --product "$PRODUCT" \
    --nproc "$nproc" \
    --fdate "$FDATE" \
    --map-type "$MAP_TYPE" \
    --base-path /discover/nobackup/"$USER"/EarthNow/plots \
    --style "$STYLE_TYPE"
  exit
fi
