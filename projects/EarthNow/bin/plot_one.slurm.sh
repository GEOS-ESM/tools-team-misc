#!/bin/bash
#SBATCH --job-name=global_vort_images
#SBATCH --output=slurm-%j.out      # Capture both stdout and stderr
#SBATCH --error=slurm-%j.out       # Same log file for errors
#SBATCH --account=s1460
#SBATCH --time=0:30:00

./plot_one.sh global all

