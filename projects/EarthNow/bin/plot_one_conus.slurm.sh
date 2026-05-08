#!/bin/bash
#SBATCH --job-name=conus_vort_images
#SBATCH --output=/discover/nobackup/%u/tools-team-misc/projects/EarthNow/output/%x-%j.out
#SBATCH --account=s1460
#SBATCH --constraint=mil
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --time=0:30:00
#SBATCH --no-requeue

./plot_one.sh conus grey_topo 20260508_00z "vorticity_heights_500mb_EarthNow" all



