source /usr/share/lmod/lmod/init/csh

module purge

module use -a /discover/swdev/gmao_SIteam/modulefiles-SLES12

module load GEOSenv
module load python/GEOSpyD/Ana2019.10_py2.7
module load nco
module load ncl

setenv GAVERSION 2.1.0.oga.1
setenv PYTHONPATH /home/dao_ops/gmao_packages/WxModel/asia-aq/lib:/home/dao_ops/gmao_packages/lib/python2.7/site-packages:/home/adasilva/src/pygrads/build/lib

setenv PATH "${PATH}:/home/dao_ops/gmao_packages/WxModel/src:/home/dao_ops/gmao_packages/WxModel/asia-aq/bin:$SHARE/dasilva/opengrads/Contents"

umask 022
