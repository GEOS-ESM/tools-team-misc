#!/bin/bash

#NOTE: Running using the uv python config

# Add input arg to run various tests
# Check if arguments are provided
if [ "$#" -lt 5 ]; then
  echo "Error: Incorrect number of arguments."
  echo "Usage: ${BASH_SOURCE[0]} <conus|global> <Style> <fdate (YYYYMMDD_HHz)> <product_name> <single|all> [-p PDATE (YYYYMMDD_HHHHz)] [-l CONSOLE_LOGLEVEL] [-f LOGFILE] [-v LOGFILE_LOGLEVEL]"
  echo "where frame time is only required if 'single' is specified, and LOGFILE is not required alongside the -f flag if you want to use the default logfile naming."
  exit 1
fi

REGION="${1,,}"
STYLE="$2"
FDATE="${3,,}"
PRODUCT="$4"
FRAMES="${5,,}"

# Shift away the 5 required positional arguments
shift 5

# Handle optional arguments
# Colon after the argument flag means that it requires a value if the flag is used
while getopts "p:l:fv:" opt; do
    case ${opt} in
        p ) PDATE="$OPTARG" ;;
        l ) CONSOLE_LEVEL="$OPTARG" ;;
        f ) 
            # Look at next arg 
            nextarg=${!OPTIND}

            # If next_arg exists and doesn't start with a '-'
            # then the user has specified a custom logfile name
            if [[ -n "$next_arg" && "$next_arg" != -* ]]; then
                LOGFILE="$next_arg"
                ((OPTIND++)) # Skip this argument in getopts loop
            else
                # Flag was provided but we'll use the default log filename
                # defined in plotall.py
                LOGFILE="USE_DEFAULT"
            fi
            ;;
        v ) LOGFILE_LEVEL="$OPTARG" ;;
        \? ) echo "Invalid option"; exit 1 ;;
    esac
done

# Argument validation
# ===================


$PDATE = "${PDATE,,}"
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

# Set console level python argument
PYTHON_CONSOLE_LOG_ARG=""

if [[ -n "$CONSOLE_LEVEL" ]]; then
    PYTHON_CONSOLE_ARG="--console level $CONSOLE_LEVEL"
fi

# Set up logfile python argument
PYTHON_LOG_ARG=""

if [[ "$LOGFILE" == "USE_DEFAULT" ]]; then
    # User typed -f but no filename. Let python use the `const`.
    PYTHON_LOG_ARG="--log-file"
elif [[ -n "$LOGFILE" ]]; then
    # User included custom filename for logs
    PYTHON_LOG_ARG="--log_file $LOGFILE"
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
    --style "$STYLE"\
    $PYTHON_LOG_ARG
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
    --style "$STYLE" \
    $PYTHON_LOG_ARG
  exit
fi
