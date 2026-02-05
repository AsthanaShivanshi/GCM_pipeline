#!/bin/bash
#SBATCH --job-name=ensmedian_taspr_rcp26
#SBATCH --output=logs/ensmedian_taspr_rcp26_job_output-%j.txt
#SBATCH --error=logs/ensmedian_taspr_rcp26_job_error-%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=15:00:00
#SBATCH --mem=256G
#SBATCH --partition=cpu
##SBATCH --partition=gpu

source ../../environment.sh

module load cdo

#  tas_day_*.nc
tas_files=(tas_day_*.nc)
if [ -e "${tas_files[0]}" ]; then
    cdo ensmedian "${tas_files[@]}" ensemble_median_tas_day_rcp26.nc
else
    echo "No tas_day_*.nc files found."
fi

#  pr_day_*.nc
pr_files=(pr_day_*.nc)
if [ -e "${pr_files[0]}" ]; then
    cdo ensmedian "${pr_files[@]}" ensemble_median_pr_day_rcp26.nc
else
    echo "No pr_day_*.nc files found."
fi