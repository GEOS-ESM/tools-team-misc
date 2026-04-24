#!/bin/bash

if [ "$#" -ne 2 ]; then
  echo "Error: Incorrect number of arguments."
  echo "Usage: $0 [conus|global] [SLURM-job-ID]" 
  exit 1
fi
MAPTYPE=$1
SLURM_JOB_ID=$2

# Access SLURM log file and extract saved image paths to file
LOG_FILE="${MAPTYPE}_vort_images-${SLURM_JOB_ID}.out"
sync
grep "Saved: " "$LOG_FILE" | awk '{print $2}' | sort -V | awk '{print "file \x27" $1 "\x27"}' > scratch/file_list.txt

PLOTS_PATH=/discover/nobackup/"$USER"/EarthNow/plots/

OUTPUT_NAME="vorticity_${MAPTYPE}_SLURM-${SLURM_JOB_ID}.mp4"
OUTPUT=$PLOTS_PATH$OUTPUT_NAME


mp4_generator.sh -o "$OUTPUT" scratch/file_list.txt

