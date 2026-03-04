#!/bin/csh -fx

#SBATCH --job-name=eta2hgt_v3.0
#SBATCH --account=s1321
#SBATCH --time=2:00:00
#SBATCH --qos=daohi
#SBATCH --ntasks=28
#SBATCH --export=NONE
#SBATCH --constraint=mil
#SBATCH --output=/discover/nobackup/dao_ops/jardizzo/FLUID/eta2hgt_v3.0_20260228_000000.log

limit stacksize unlimited
cd /home/dao_ops/jardizzo/FLUID/we-can/utils/eta2hgt_v3.0_sles15
source modules

eta2hgt.sh 20260228 000000 10 3
