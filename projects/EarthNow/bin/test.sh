#!/bin/bash

source /usr/share/lmod/lmod/init/bash
module load python/GEOSpyD
module load ffmpeg

python plotall.py \
    --product max_reflectivity \
    --nproc 1 \
    --fdate 20260202_00z \
    --map-type conus \
    --base-path /discover/nobackup/jardizzo/EarthNow/plots \
    --map-type goes_east_full_disk \
    --style satellite
exit 0
#   --pdate 20260202_1600 \
