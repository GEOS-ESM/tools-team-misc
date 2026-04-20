#!/bin/bash
#SBATCH --job-name=conus_vort_image_single
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --account=s1460
#SBATCH --time=0:30:00
#SBATCH --no-requeue

./plot_one.sh conus single




