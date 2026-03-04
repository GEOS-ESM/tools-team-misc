#!/bin/sh

if [ $# -ne 4 ]; then
  echo "Usage: $0 [ccyymmdd] [hhmmss] [flen] [hinc]"
  exit 1
else
  idate=$1
  itime=$2
  flen=`expr $3 + 100 | cut -c2,3`
  hinc=$4
fi

fcst_dt=`timetag $idate $itime "{%Y%m%dT%H%M%S}%+d00"`
start_dt=$fcst_dt
if [ $flen -eq 30 ]; then
  end_dt=`timetag $idate $itime "{%Y%m%dT%H%M%S}%+H$flen"`
else
  end_dt=`timetag $idate $itime "{%Y%m%dT%H%M%S}%+d$flen"`
fi
yyyymmddhh=`timetag $idate $itime "{%Y %m %d %H}%+d00"`

DEST=/discover/nobackup/projects/gmao/merra2/data/pub/supplemental/fp_Np

# *********************
# Create Np collections
# *********************

# inst3_3d_aer_Nv 
# ===============

VARS="PM25,PM10,PM,DUST,BCOC"
LEVELS="1000,975,950,925,900,850,800,750,700,650,600,550,500,450,400,350,300,250,200,150,100,70,50,30,20,10"

NVFILE=/discover/nobackup/projects/gmao/gmao_ops/pub/fp/forecast/Y%%Y/M%%m/D%%d/H%%H/GEOS.fp.fcst.inst3_3d_aer_Nv.%%Y%%m%%d_%%H+%Y%m%d_%H00.V01.nc4
NPFILE=$DEST/forecast/Y%%Y/M%%m/D%%d/H%%H/GEOS.fp.fcst.inst3_3d_aer_Np

eta2prs.py --iname $NVFILE --oname $NPFILE \
           --levels "$LEVELS" \
           --vars "$VARS" \
           --fcst_dt $fcst_dt \
           --start_dt $start_dt \
           --end_dt  $end_dt \
           --t_deltat $hinc

# ********************************
# Create the Opendap control files
# ********************************

g5_ddf.pl -c -v -d $DEST $DEST $yyyymmddhh

files=`find $DEST/opendap/ -type f ! -name "*.portal"`

for pathname in $files; do

  if [ ! -f $pathname.portal ]; then
#   cat $pathname | sed 's@data/pub/supplemental@pub/supplemental@g' > $pathname.portal
    cat $pathname | sed s'@dset ^../../../@dset /discover/nobackup/projects/gmao/merra2/pub/supplemental/fp_Np@g' > $pathname.portal

  fi

done

exit 0
