#!/bin/bash

source /usr/share/lmod/lmod/init/bash
module load python/GEOSpyD
module load ffmpeg

bindir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
rootdir="$(dirname "$bindir")"
srcdir="$rootdir/src"
listing="$bindir/listing"


PYTHONPATH="$srcdir${PYTHONPATH:+:$PYTHONPATH}" \
    python "$bindir/plotall.py" \
    --product carbon_EarthNow \
    --nproc 1 \
    --fdate 20260202_00z \
    --pdate 20260202_1600 \
    --map-type global \
    --base-path /discover/nobackup/$USER/EarthNow/plots \
    --style light \
    --boundaries countries \
    --boundaries states 
exit 0
