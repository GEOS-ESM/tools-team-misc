#!/bin/bash

#NOTE: Getting rid of this to test the uv venv
# source /usr/share/lmod/lmod/init/bash
# module load python/GEOSpyD
# module load ffmpeg

bindir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
rootdir="$(dirname "$bindir")"
srcdir="$rootdir/src"
listing="$bindir/listing"


# Test GLOBAL
PYTHONPATH="$srcdir${PYTHONPATH:+:$PYTHONPATH}" \
    python "$bindir/plotall.py" \
    --product temperature_2m_EarthNow \
    --nproc 1 \
    --fdate 20260202_00z \
    --pdate 20260202_1600 \
    --map-type global \
    --base-path $NOBACKUP/EarthNow/plots \
    --style light \
    --boundaries countries coastlines
exit 0

# Test CONUS
# PYTHONPATH="$srcdir${PYTHONPATH:+:$PYTHONPATH}" \
#     python "$bindir/plotall.py" \
#     --product temperature_2m_EarthNow \
#     --nproc 1 \
#     --fdate 20260202_00z \
#     --pdate 20260202_1600 \
#     --map-type conus \
#     --base-path $NOBACKUP/EarthNow/plots \
#     --style light \
#     --boundaries countries \
#     --boundaries states \
#     --station_values
# exit 0

