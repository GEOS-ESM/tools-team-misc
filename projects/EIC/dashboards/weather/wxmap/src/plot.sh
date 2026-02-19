#!/bin/sh

if [ $# -ne 2 ]; then
  echo "Usage: $0 [sccyymmdd] [eccyymmdd]" 2>&1
  exit 1
fi

sdate=$1
edate=$2

source /discover/nobackup/jardizzo/Shared/WxMap/utils/modules.bash

wxmap.py --config ../eic \
         --config ../eic_analysis \
         --stream GEOS \
         --start_dt $sdate \
         --end_dt $edate \
         --t_deltat PT1H \
         --field q2m \
         --region global \
         --fullframe --lights_off --no_title --no_label --no_logo \
         --geometry 4096x2048 \
         --oname EIC_weather.nasa.gmao.geos-fp.analysis.q2m.Colorbar_12.4096x2048.%Y%m%d%H%M.png
