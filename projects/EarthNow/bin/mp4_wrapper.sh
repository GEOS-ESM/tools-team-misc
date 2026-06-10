#!/bin/bash

if [ "$#" -ne 2 ]; then
  echo "Error: Incorrect number of arguments."
  echo "Usage: $0 [conus|global] [SLURM-job-ID]" 
  exit 1
fi
MAPTYPE=$1
SLURM_JOB_ID=$2

# Set absolute path to work in
OUTPUT_PATH="/discover/nobackup/$USER/tools-team-misc/projects/EarthNow/output/"

# Access SLURM log file and extract saved image paths to file
LIST_PATH="$OUTPUT_PATH"file_list.txt
LOG_FILE="$OUTPUT_PATH${MAPTYPE}_vort_images-${SLURM_JOB_ID}.out"
sync
grep "Saved: " "$LOG_FILE" | awk '{print $2}' | sort -V | awk '{print "file \x27" $1 "\x27"}' > "$LIST_PATH"


PLOTS_PATH=/discover/nobackup/"$USER"/EarthNow/plots/

OUTPUT_NAME="vorticity_${MAPTYPE}_SLURM-${SLURM_JOB_ID}.mp4"
OUTPUT=$PLOTS_PATH$OUTPUT_NAME


mp4_generator.sh -o "$OUTPUT" "$LIST_PATH"

