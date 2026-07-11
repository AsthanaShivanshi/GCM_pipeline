#!/bin/bash
#SBATCH --job-name=UNet_allcells
#SBATCH --output=logs/bcsr/UNet_allcells_output-%A_%a.log
#SBATCH --error=logs/bcsr/UNet_allcells_job_error-%A_%a.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=18:00:00
#SBATCH --mem=128G
#SBATCH --partition=gpu-l40   ##OR gpu-h100
#SBATCH --array=0-11 ##12 blocks for 10 years each 
#SBATCH --gres=gpu:1

module load python
source diffscaler.sh
module load cdo

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

START_YEARS=(1981 1991 2001 2011 2021 2031 2041 2051 2061 2071 2081 2091)
END_YEARS=( 1990 2000 2010 2020 2030 2040 2050 2060 2070 2080 2090 2100)

START_YEAR=${START_YEARS[$SLURM_ARRAY_TASK_ID]}
END_YEAR=${END_YEARS[$SLURM_ARRAY_TASK_ID]}

MODE=${MODE:-unet} #modes possible : bilinear, unet, ......ddim is in a separate .sh script...,,.. should be run sequentially. 


for ENSEMBLE in EQM_C CDF-t dOTC; do
    echo "($START_YEAR-$END_YEAR) started in mode $MODE for ensemble $ENSEMBLE"
    python SR_downscaling/inference_allframes_eta0_ssp370.py --start_year $START_YEAR --end_year $END_YEAR --mode $MODE --ensemble $ENSEMBLE
    echo "$MODE for ssp370 ($START_YEAR-$END_YEAR) finished for $ENSEMBLE"
done



