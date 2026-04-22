#!/bin/bash

MAPTYPE="conus"
SLURM_JOB_ID="55492940"

# Access SLURM log file and extract saved image paths to file
LOG_FILE="${MAPTYPE}_vort_images-${SLURM_JOB_ID}.out"
sync
grep "Saved: " "$LOG_FILE" | awk '{print $2}' | sort -V | awk '{print "file \x27" $1 "\x27"}' > file_list.txt

PLOTS_PATH=/discover/nobackup/"$USER"/EarthNow/plots/

OUTPUT_NAME="vorticity_${MAPTYPE}_SLURM-${SLURM_JOB_ID}.mp4"
OUTPUT=$PLOTS_PATH$OUTPUT_NAME


mp4_generator.sh -o "$OUTPUT" file_list.txt

