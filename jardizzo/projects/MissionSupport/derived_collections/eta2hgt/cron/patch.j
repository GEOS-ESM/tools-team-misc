#!/bin/csh -fx

#SBATCH --job-name=eta2hgt_v3.0
#SBATCH --account=s1321
#SBATCH --time=1:50:00
#SBATCH --qos=daohi
#SBATCH --ntasks=28
#SBATCH --export=NONE
#SBATCH --constraint=hasw
#SBATCH --output=/discover/nobackup/dao_ops/jardizzo/FLUID/eta2hgt_v3.0_20210902_000000.log

limit stacksize unlimited
cd /home/dao_ops/jardizzo/FLUID/we-can/utils/eta2hgt_v3.0
source modules

eta2hgt.sh 20210902 0 10 3
