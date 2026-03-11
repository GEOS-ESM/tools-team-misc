#!/bin/bash

source /usr/share/lmod/lmod/init/bash
module load python/GEOSpyD
module load ffmpeg

bindir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
rootdir="$(dirname "$bindir")"
srcdir="$rootdir/src"

echo $PYTHONPATH

for product in `cat $bindir/listing`; do

  PYTHONPATH="$srcdir${PYTHONPATH:+:$PYTHONPATH}" \
    python "$bindir/plotall.py" \
      --product $product \
      --nproc 1 \
      --fdate 20260202_00z \
      --pdate 20260202_1600 \
      --map-type global \
      --base-path /discover/nobackup/$USER/EarthNow/plots \
      --style light

   echo "=====> $product :  $?"

done

exit 0
