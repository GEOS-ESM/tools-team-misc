#!/bin/bash

#NOTE: Running using the uv python config

# Add input arg to run various tests
# Check if arguments are provided
if [ "$#" -ne 7 ]; then
  echo "Error: Incorrect number of arguments."
  echo "Usage: ${BASH_SOURCE[0]} <conus|global> <Style> <YYYYMMDD> <HHz> <product_name> <single|all> <nproc>"
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
FDATE=$3
if [[ ! "$FDATE" =~ ^[0-9]{8}$ ]]; then
  echo "Error: Invalid date format (YYYYMMDD)"
  exit 1
fi


TIMEZ="${4,,}"
if [[ ! $TIMEZ =~ ^[0-9]{2}z$ ]]; then
    echo "Error: Invalid time format. Use (HHz or HHZ) where HH is hours GMT"
    exit 1
fi

PRODUCT=$5

FRAMES="${6,,}"
if [[ $FRAMES != "single" && $FRAMES != "all" ]]; then
  echo "Error: Invalid frame argument. Valid args: 'single',  'all' (frames)."
  exit 1
fi

NPROC=$7
if [[ ! "$NPROC" =~ ^[0-9]+$ ]]; then
  echo "Error: Invalid nproc argument. Must contain only numbers."
  exit 1
fi


bindir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "$FRAMES" = "single" ]]; then
  # Generate single plot
  uv run "$bindir/plotall.py" \
    --product "$PRODUCT" \
    --nproc "$NPROC" \
    --fdate "${FDATE}_${TIMEZ}" \
    --pdate "${FDATE}_${TIMEZ}" \
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
    --nproc "$NPROC" \
    --fdate "${FDATE}_${TIMEZ}" \
    --map-type "$MAP_TYPE" \
    --base-path /discover/nobackup/"$USER"/EarthNow/plots \
    --style "$STYLE_TYPE"
  exit
fi
