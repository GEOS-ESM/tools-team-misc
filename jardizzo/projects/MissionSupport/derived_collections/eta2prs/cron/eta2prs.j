#!/bin/csh -fx

#SBATCH --job-name=eta2prs_v1.0
#SBATCH --account=g2538
#SBATCH --time=1:30:00
#SBATCH --qos=daohi
#SBATCH --ntasks=28
#SBATCH --export=NONE
#SBATCH --constraint=mil
#SBATCH --output=/discover/nobackup/dao_ops/jardizzo/FLUID/eta2prs_v1.0_20260228_000000.log

limit stacksize unlimited
cd /home/dao_ops/jardizzo/FLUID/we-can/utils/eta2prs_v1.0_sles15
source modules

eta2prs.sh 20260228 000000 10 3
