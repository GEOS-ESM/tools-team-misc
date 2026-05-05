#!/bin/bash

#NOTE: Running using the uv python config

# Add input arg to run various tests
# Check if arguments are provided
if [ "$#" -lt 5 ] || [ "$#" -gt 6 ]; then
  echo "Error: Incorrect number of arguments."
  echo "Usage: ${BASH_SOURCE[0]} <conus|global> <Style> <forecast time as YYYYMMDD_HHz> <product_name> <single|all> [frame time as YYYYMMDD_HHHHz]"
  echo "where frame time is only required if 'single' is specified"
  exit 1
fi

REGION="${1,,}"
STYLE="$2"
FDATE="${3,,}"
PRODUCT="$4"
FRAMES="${5,,}"
PDATE="${6,,}"

# Argument validation
# ===================

# If "single," then we MUST also have a PDATE
if [[ "$FRAMES" == "single" ]]; then 
    if [ -z "$PDATE" ]; then 
        echo "Error: If 'single' is specified, you must provide the PDATE (YYYYMMDD_HHHHz) for the desired frame."
        exit 1
    fi

    # PDATE defines the frame/timestamp
    if [[ ! $PDATE =~ ^[0-9]{8}_[0-9]{4}z$ ]]; then
        echo "Error: Invalid PDATE format. Must be YYYYMMDD_HHHHz"
    exit 1
    fi
elif [[ "$FRAMES" == "all" ]]; then 
    if [ -n "$PDATE" ]; then 
        echo "Note: PDATE ($PDATE) not in use when 'all' is specified."
    fi
else
    echo: "Error: Invalid argument 5. Valid args: 'single', 'all'."
    exit 1
fi

# Parse region
if [[ $REGION != "conus" && $REGION != "global" ]]; then
  echo "Error: Invalid region argument. Valid args: 'conus',  'global'."
  exit 1
fi

# Parse date
# FDATE defines the *directory* (forecast initialization time) 
if [[ ! $FDATE =~ ^[0-9]{8}_[0-9]{2}z$ ]]; then
  echo "Error: Invalid first date format (YYYYMMDD_HHz)"
  exit 1
fi

# Call plotall.py 
# ===================
#
bindir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "$FRAMES" = "single" ]]; then
  # Generate single plot
  uv run "$bindir/plotall.py" \
    --product "$PRODUCT" \
    --nproc 1 \
    --fdate "$FDATE" \
    --pdate "$PDATE" \
    --map-type "$REGION" \
    --base-path /discover/nobackup/"$USER"/EarthNow/plots \
    --style "$STYLE"
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
    --map-type "$REGION" \
    --base-path /discover/nobackup/"$USER"/EarthNow/plots \
    --style "$STYLE"
  exit
fi
