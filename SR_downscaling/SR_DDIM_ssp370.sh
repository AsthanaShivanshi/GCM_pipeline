#!/bin/bash
#SBATCH --job-name=hist_DDIM_allcells
#SBATCH --output=logs/SR/hist_DDIM_allcells_output-%A_%a.log
#SBATCH --error=logs/SR/hist_DDIM_allcells_job_error-%A_%a.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=15:00:00
#SBATCH --mem=128G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
##SBATCH --array=0-118  ## total tbp::: 120 years: 1981-2099
#SBATCH --array=0-42  ## 43 years: 1981-2023,, submitted to gpu
##SBATCH --array=43-118  ## subsequently tbs :: 43-118 2024-2099, submitted to l40


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

for ENSEMBLE in EQM_C CDF-t dOTC; do
    echo "Year $YEAR started in mode $MODE for ensemble $ENSEMBLE"
    python SR_downscaling/inference_allframes_eta0_ssp370.py --start_year $YEAR --end_year $YEAR --mode $MODE --ensemble $ENSEMBLE
    echo "$MODE for ssp370 year $YEAR finished for $ENSEMBLE"
done