#!/bin/bash
#SBATCH --job-name=conus_vort_images
#SBATCH --output=scratch/%x-%j.out
#SBATCH --account=s1460
#SBATCH --constraint=mil
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --time=0:30:00
#SBATCH --no-requeue

./plot_one.sh conus 20260401 "vorticity_heights_500mb_EarthNow" single



