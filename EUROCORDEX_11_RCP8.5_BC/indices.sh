#!/bin/bash
#SBATCH --job-name=benchmarking_pss_climatology_rcp85
#SBATCH --output=logs/pss_climatology_rcp85_job-%j.log
#SBATCH --error=logs/pss_climatology_rcp85_job-%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1-00:00:00
#SBATCH --mem=256G
#SBATCH --partition=cpu
##SBATCH --partition=gpu
module load cdo
source environment.sh
cd /work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/GCM_pipeline/EUROCORDEX_11_RCP8.5_BC

#python daily_climatology_bc_winner.py


python benchmarking.py
