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

for product in `cat $bindir/listing`; do

  python plotall.py \
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
