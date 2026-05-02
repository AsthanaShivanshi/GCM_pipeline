#!/bin/bash
#SBATCH --job-name=pdf_log_precip_job
#SBATCH --output=logs/logpdf_log_precip_job-%j.log
#SBATCH --error=logs/logpdf_log_precip_job-%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --mem=256G
#SBATCH --partition=cpu
##SBATCH --partition=gpu


source environment.sh


cd /work/FAC/FGSE/IDYST/tbeucler/downscaling/sasthana/Downscaling/GCM_pipeline


python EUROCORDEX_11_RCP8.5_BC/Metrics/pooled_bar_plots_precip.py

#python EUROCORDEX_11_RCP8.5_BC/Metrics/pooled_bar_plots_temp.py


