#!/bin/bash

source /usr/share/lmod/lmod/init/bash
module load python/GEOSpyD
module load ffmpeg

bindir=`dirname $0`
cd $bindir
bindir=`pwd`
rootdir=`dirname $bindir`
srcdir=$rootdir/src

if [ -z "$PYTHONPATH" ]; then
  export PYTHONPATH=$srcdir
else
  export PYTHONPATH=${PYTHONPATH}:$srcdir
fi

echo $PYTHONPATH

python plotall.py \
    --product max_reflectivity \
    --nproc 1 \
    --fdate 20260202_00z \
    --pdate 20260202_1600 \
    --map-type conus \
    --base-path /discover/nobackup/$USER/EarthNow/plots \
    --map-type goes_east_full_disk \
    --style satellite
exit 0
