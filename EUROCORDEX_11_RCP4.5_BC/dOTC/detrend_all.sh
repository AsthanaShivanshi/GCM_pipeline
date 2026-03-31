#!/bin/bash
#SBATCH --job-name=detrending_taspr_dOTC_rcp45
#SBATCH --output=logs/detrending_taspr_dOTC_rcp45_job_output-%j.txt
#SBATCH --error=logs/detrending_taspr_dOTC_rcp45_job_error-%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=15:00:00
#SBATCH --mem=256G
#SBATCH --partition=cpu
##SBATCH --partition=gpu

module load cdo
source environment.sh

OUT_DIR="/work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/GCM_pipeline/EUROCORDEX_11_RCP4.5_BC_detrended/dOTC/"
cd /work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/GCM_pipeline/EUROCORDEX_11_RCP4.5_BC/dOTC/



#detrending

tas_files=(tas_day_EUR-11_*.nc)

if [ -e "${tas_files[0]}" ]; then #checking if the first element (file) exists
    for file in "${tas_files[@]}"; do
        base="${file%.nc}"  



        cdo detrend "${file}"  "${OUT_DIR}${base}_detrended.nc"

        cdo trend "${file}"  "${OUT_DIR}${base}_slope.nc" "${OUT_DIR}${base}_intercept.nc"
    done
fi



pr_files=(pr_day_EUR-11_*.nc)

if [ -e "${pr_files[0]}" ]; then #checking if the first element (file) exists
    for file in "${pr_files[@]}"; do
        base="${file%.nc}"  



        cdo detrend "${file}"  "${OUT_DIR}${base}_detrended.nc"

        cdo trend "${file}"  "${OUT_DIR}${base}_slope.nc" "${OUT_DIR}${base}_intercept.nc"
    done
fi