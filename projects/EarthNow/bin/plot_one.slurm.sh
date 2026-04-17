#!/bin/bash
#SBATCH --job-name=global_vort_images
#SBATCH --constraint=mil
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#SBATCH --account=s1460
#SBATCH --time=0:30:00
#SBATCH --no-requeue

./plot_one.sh global all

