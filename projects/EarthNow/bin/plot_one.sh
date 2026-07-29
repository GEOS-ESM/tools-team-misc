#!/bin/bash

# 5/12/26 Recent changes to input args: see example syntax below: 
# plot_one.sh global grey_topo 20260508_00z "vorticity_heights_500mb_EarthNow" single -p  20260508_0000z  -l "INFO"

# Usage function for clean error handling
usage() {
  echo "Usage: ${BASH_SOURCE[0]} <conus|global> <Style> <forecast time> <product_name> <single|all> [OPTIONS]"
  echo ""
  echo "Required arguments:"
  echo "  1. Region         : conus or global"
  echo "  2. Style          : Style name"
  echo "  3. Forecast time  : YYYYMMDD_HHz"
  echo "  4. Product name   : Name of the product"
  echo "  5. Frames           : single or all"
  echo ""
  echo "Optional arguments:"
  echo "  -p, --pdate: Frame time as YYYYMMDD_HHHHz (Required if Mode is 'single')"
  echo "  -r, --data-reader: Name of reader to use, specific to data stream"
  echo "  -l, --logger-opt: Logger option (e.g., DEBUG, INFO)"
  echo "  -b, --boundaries: List of boundaries to plot"
  echo "  -h, --help: Display this help message"
  exit 1
}

# Check if we have at least the 5 required arguments
if [ "$#" -lt 5 ]; then
  echo "!! Error: Missing required arguments."
  usage
fi

# Assign required positional arguments
REGION="${1,,}"
STYLE="$2"
FDATE="${3,,}"
PRODUCT="$4"
FRAMES="${5,,}"

# Shift the first 5 arguments out of the way so we can parse the optionals
shift 5

# Parse remaining optional arguments using a while/case loop
while [[ "$#" -gt 0 ]]; do
  case $1 in
    -p|--pdate)
      PDATE="${2,,}"
      shift 2 # Shift past the flag and its value
      ;;
    -r|--data-reader)
      READER=$2
      shift 2
      ;;
    -l|--logger-opt)
      LOGGER="$2"
      shift 2
      ;;
    -b|--boundaries)
      BOUNDARIES_LIST=()
      # Check if a list follows, then loop to grab everything until the next flag
      shift
      while [[ $# -gt 0 && ! "$1" =~ ^- ]]; do
        BOUNDARIES_LIST+=("$1")
        shift
      done
      ;;
    -h|--help)
      usage
      ;;
    *)
      echo "Error: Unknown parameter passed: $1"
      usage
      ;;
  esac
done


# Argument validation
# ===================

# If "single," then we MUST also have a PDATE
if [[ "$FRAMES" == "single" ]]; then 
    if [ -z "$PDATE" ]; then 
        echo "Error: If 'single' is specified, you must provide the PDATE (YYYYMMDD_HHHHz) for the desired frame."
        exit 1
    fi

    # PDATE defines the frame/timestamp
    if [[ ! $PDATE =~ ^[0-9]{8}_[0-9]{2}z$ ]]; then
        echo "Error: Invalid PDATE format. Must be YYYYMMDD_HHz"
    exit 1
    fi
elif [[ "$FRAMES" == "all" ]]; then 
    if [ -n "$PDATE" ]; then 
        echo "Note: PDATE ($PDATE) not in use when 'all' is specified and is ignored."
        unset PDATE
    fi
else
    echo "Error: Invalid argument 5. Valid args: 'single', 'all'."
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

# Create Python command
# ===================
# Determine the number of processors to use based on SLURM environment variable, if running login, default to 1
 if  [[ -n "$SLURM_CPUS_PER_TASK" ]]; then
    NPROC="$SLURM_CPUS_PER_TASK"
  else
    NPROC=1
  fi

bindir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_CMD=(
  uv run "$bindir/plotall.py"
    --product "$PRODUCT"
    --nproc "$NPROC" 
    --fdate "$FDATE"
    --map-type "$REGION"
    --base-path /discover/nobackup/"$USER"/EarthNow/plots
    --style "$STYLE"
)
# Optional arguments
if [[ -n "$PDATE" ]]; then
  # Append the flag and the value to the array
  PYTHON_CMD+=(--pdate "$PDATE")
fi

if [[ -n "$READER" ]]; then
  PYTHON_CMD+=(--data-reader "$READER")
fi

if [[ -n "$LOGGER" ]]; then
  PYTHON_CMD+=(--logger "$LOGGER")
fi

if [[ -n "$BOUNDARIES_LIST" ]]; then
  PYTHON_CMD+=(--boundaries "${BOUNDARIES_LIST[@]}")
fi

# Execute the array
# ===================
echo "Executing: ${PYTHON_CMD[*]}"
"${PYTHON_CMD[@]}"
