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

DEST=/discover/nobackup/projects/gmao/merra2/data/pub/supplemental

# *********************
# Create Nz collections
# *********************

# Define Height Levels and ETA file
# containing vertical coordinate information
# ==========================================

#LEVELS="0,500,1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,6000,6500,7000,7500,8000,8500,9000,9500,10000"
LEVELS="0,50,100,200,300,400,500,1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,6000,6500,7000,7500,8000,8500,9000,9500,10000"
HNAME=/discover/nobackup/projects/gmao/gmao_ops/pub/fp/forecast/Y%%Y/M%%m/D%%d/H%%H/GEOS.fp.fcst.inst3_3d_asm_Nv.%%Y%%m%%d_%%H+%Y%m%d_%H00.V01.nc4

# inst3_3d_aer_Nv
# ===============

VARS="BCOC,PM25,PM,DUST"

INAME=/discover/nobackup/projects/gmao/gmao_ops/pub/fp/forecast/Y%%Y/M%%m/D%%d/H%%H/GEOS.fp.fcst.inst3_3d_aer_Nv.%%Y%%m%%d_%%H+%Y%m%d_%H00.V01.nc4
ONAME=$DEST/fp/forecast/Y%%Y/M%%m/D%%d/H%%H/GEOS.fp.fcst.inst3_3d_aer_Nz.%%Y%%m%%d_%%H+%Y%m%d_%H00.V01.nc4

eta2hgt_iter.py --iname $INAME --hname $HNAME --oname $ONAME \
                --vars $VARS --levels $LEVELS \
                --fcst_dt $fcst_dt \
                --start_dt $start_dt \
                --end_dt  $end_dt \
                --t_deltat $hinc \
                --strict

# inst3_3d_asm_Nv
# ===============

VARS="QL,QI,U,V,RH,THETA,AGL"

INAME=/discover/nobackup/projects/gmao/gmao_ops/pub/fp/forecast/Y%%Y/M%%m/D%%d/H%%H/GEOS.fp.fcst.inst3_3d_asm_Nv.%%Y%%m%%d_%%H+%Y%m%d_%H00.V01.nc4
ONAME=$DEST/fp/forecast/Y%%Y/M%%m/D%%d/H%%H/GEOS.fp.fcst.inst3_3d_asm_Nz.%%Y%%m%%d_%%H+%Y%m%d_%H00.V01.nc4

eta2hgt_iter.py --iname $INAME --hname $HNAME --oname $ONAME \
                --vars $VARS --levels $LEVELS \
                --fcst_dt $fcst_dt \
                --start_dt $start_dt \
                --end_dt  $end_dt \
                --t_deltat $hinc \
                --strict

# *********************
# Create Nh collections
# *********************

# Define Height Levels and ETA file
# containing vertical coordinate information
# ==========================================

LEVELS="1000,2000,4000,6000"
HNAME=/discover/nobackup/projects/gmao/gmao_ops/pub/fp/forecast/Y%%Y/M%%m/D%%d/H%%H/GEOS.fp.fcst.inst3_3d_asm_Nv.%%Y%%m%%d_%%H+%Y%m%d_%H00.V01.nc4

# inst3_3d_aer_Nv
# ===============

VARS="BCOC,PM25,PM,DUST"

INAME=/discover/nobackup/projects/gmao/gmao_ops/pub/fp/forecast/Y%%Y/M%%m/D%%d/H%%H/GEOS.fp.fcst.inst3_3d_aer_Nv.%%Y%%m%%d_%%H+%Y%m%d_%H00.V01.nc4
ONAME=$DEST/fp/forecast/Y%%Y/M%%m/D%%d/H%%H/GEOS.fp.fcst.inst3_3d_aer_Nh.%%Y%%m%%d_%%H+%Y%m%d_%H00.V01.nc4

eta2hgt_iter.py --iname $INAME --hname $HNAME --oname $ONAME \
                --vars $VARS --levels $LEVELS \
                --fcst_dt $fcst_dt \
                --start_dt $start_dt \
                --end_dt  $end_dt \
                --t_deltat $hinc \
                --strict --feet --ground --alt

# inst3_3d_asm_Nv
# ===============

VARS="QL,QI,U,V,RH,THETA"

INAME=/discover/nobackup/projects/gmao/gmao_ops/pub/fp/forecast/Y%%Y/M%%m/D%%d/H%%H/GEOS.fp.fcst.inst3_3d_asm_Nv.%%Y%%m%%d_%%H+%Y%m%d_%H00.V01.nc4
ONAME=$DEST/fp/forecast/Y%%Y/M%%m/D%%d/H%%H/GEOS.fp.fcst.inst3_3d_asm_Nh.%%Y%%m%%d_%%H+%Y%m%d_%H00.V01.nc4

eta2hgt_iter.py --iname $INAME --hname $HNAME --oname $ONAME \
                --vars $VARS --levels $LEVELS \
                --fcst_dt $fcst_dt \
                --start_dt $start_dt \
                --end_dt  $end_dt \
                --t_deltat $hinc \
                --strict --feet --ground --alt

# ********************************
# Create the Opendap control files
# ********************************

#g5_ddf.pl -c -v -d $DEST $DEST $yyyymmddhh

/home/dao_ops/GEOSadas-CURRENT/GEOSadas/Linux/bin_ops/g5_ddf.pl -c -v \
           -d $DEST/fp $DEST/fp $yyyymmddhh

COLLECTIONS="inst3_3d_aer_Nz inst3_3d_asm_Nz inst3_3d_aer_Nh inst3_3d_asm_Nh"

tnode=`timetag $idate $itime "{%Y%m%d_%H}%+d00"`

for collection in $COLLECTIONS; do
    file=$DEST/fp/opendap/fcast/$collection/$collection.$tnode
    cat $file | sed 's@data/pub/supplemental@pub/supplemental@g' > $file.portal

done

exit 0
