#!/bin/bash
#SBATCH --job-name=dOTC_missed_proj_DDIM_allcells
#SBATCH --output=logs/SR/dOTC_missed_proj_DDIM_allcells_output-%A_%a.log
#SBATCH --error=logs/SR/dOTC_missed_proj_DDIM_allcells_job_error-%A_%a.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=10:00:00
#SBATCH --mem=128G
#SBATCH --partition=gpu-l40
#SBATCH --gres=gpu:1
##SBATCH --array=31-42  ## old partial runs : entire ref period
#SBATCH --array=34-118  ## 2015-2099 ; missed dOTC runs 


#¨¨¨¨¨¨!!!!!!!!This script to be executed only after SR_UNet_xx.sh is complete for both bicubic and unet modes.
#AstahanSh.

module load python
source diffscaler.sh
module load cdo


cd ../GCM_pipeline

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK


YEAR=$((1981 + SLURM_ARRAY_TASK_ID))  ## one year/job
MODE=${MODE:-ddim}



#EQM_C CDF-t dOTC

for ENSEMBLE in dOTC; do
    echo "Year $YEAR started in mode $MODE for ensemble $ENSEMBLE"
    python SR_downscaling/inference_allframes_eta0_ssp370.py --start_year $YEAR --end_year $YEAR --mode $MODE --ensemble $ENSEMBLE
    echo "$MODE for ssp370 year $YEAR finished for $ENSEMBLE"
done