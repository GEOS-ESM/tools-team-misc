#!/bin/sh

if [ $# -ne 2 ]; then
  echo "Usage: $0 [ccyymmdd] [hhmmss]"
  exit 1
else
  idate=$1
  itime=$2
fi

hour=`expr $itime / 10000`
if [ $hour -eq 0 ]; then
  flen=10
else
  if [$hour -eq 12 ]; then
    flen=5
  else
    flen=30
  fi
fi


echo $flen
