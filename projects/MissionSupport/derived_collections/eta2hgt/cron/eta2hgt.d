#!/bin/sh

if [ $# -ne 2 ]; then
  echo "Usage: $0 [ccyymmdd] [hhmmss]"
  exit 1
else
  idate=$1
  itime=$2
fi

exec >/dev/null 2>&1

SENTINEL=/discover/nobackup/projects/gmao/yotc/pub/fp/opendap/fcast/inst1_2d_hwl_Nx/inst1_2d_hwl_Nx.%Y%m%d_%H
#SENTINEL=/discover/nobackup/projects/gmao/gmao_ops/pub/f516_fp/forecast/Y%Y/M%m/D%d/H%H/GEOS.fp.fcst.inst1_2d_hwl_Nx.%Y%m%d_%H+%%Y%%m%%d_%%H%%M.V01.nc4

while [ 1 ]; do

  hour=`expr $itime / 10000`
  if [ $hour -eq 0 ]; then
    flen=10
  else
    if [ $hour -eq 12 ]; then
      flen=5
    else
      flen=30
    fi
  fi

  if [ $flen -eq 30 ]; then
    edattim=`timetag $idate $itime {%Y%m%d_%H%M%S}%+H$flen`
  else
    edattim=`timetag $idate $itime {%Y%m%d_%H%M%S}%+d$flen`
  fi
  edate=`echo $edattim | cut -d'_' -f1`
  etime=`echo $edattim | cut -d'_' -f2`
  sentinel=`timetag $idate $itime $SENTINEL`
  sentinel=`timetag $edate $etime $sentinel`

  if [ -f $sentinel ]; then

    sleep 300

    cat <<EOF | sed -n '1,$s/^ *//p' > eta2hgt.j
    #!/bin/csh -fx

    #SBATCH --job-name=eta2hgt_v3.0
    #SBATCH --account=s1321
    #SBATCH --time=2:00:00
    #SBATCH --qos=daohi
    #SBATCH --ntasks=28
    #SBATCH --export=NONE
    #SBATCH --constraint=mil
    #SBATCH --output=/discover/nobackup/dao_ops/jardizzo/FLUID/eta2hgt_v3.0_${idate}_${itime}.log

    limit stacksize unlimited
    cd /home/dao_ops/jardizzo/FLUID/we-can/utils/eta2hgt_v3.0_sles15
    source modules

    eta2hgt.sh $idate $itime $flen 3
EOF

    sbatch eta2hgt.j

    dattim=`timetag $idate $itime {%Y%m%d_%H%M%S}%+H06`
    idate=`echo $dattim | cut -d'_' -f1`
    itime=`echo $dattim | cut -d'_' -f2`

  else

    sleep 300

  fi

done

exit 0
