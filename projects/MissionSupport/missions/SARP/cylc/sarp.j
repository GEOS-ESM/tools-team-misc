#!/bin/csh

#SBATCH --job-name=WxSARP
#SBATCH --account=g2538
#SBATCH --time=1:15:00
#SBATCH --qos=daohi
#SBATCH --ntasks=28
#SBATCH --export=NONE
#SBATCH --constraint=mil
#SBATCH --output=/discover/nobackup/dao_ops/jardizzo/FLUID/sarp.log

module purge
module use -a /discover/swdev/gmao_SIteam/modulefiles-SLES15

module load GEOSenv
module load comp/gcc/11.4.0
module load comp/intel/2021.6.0
module load mpi/impi/2021.13
module load ffmpeg/5.0
module load ImageMagick
module load python/GEOSpyD/Min24.4.0-0_py3.11

setenv GAVERSION 2.1.0.oga.1
setenv PYTHONPATH /discover/nobackup/jardizzo/Shared/WxMap_v2.0/lib:/home/dao_ops/cylc8-workflows/SARP/lib

setenv PATH "${PATH}:/discover/nobackup/jardizzo/Shared/WxMap_v2.0/utils:/home/dao_ops/cylc8-workflows/SARP/bin:$SHARE/dasilva/opengrads/Contents"

umask 022

#sarp_plot.py 20260503T00 sarp-cf.yml
sarp_plot.py 20260504T00 sarp-fp.yml
