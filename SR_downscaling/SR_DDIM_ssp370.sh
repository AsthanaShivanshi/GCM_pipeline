#!/bin/bash
#SBATCH --job-name=Model_DDIM_allcells
#SBATCH --output=logs/SR/Model_DDIM_allcells_output-%A_%a.log
#SBATCH --error=logs/SR/Model_DDIM_allcells_job_error-%A_%a.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01-00:00:00
#SBATCH --mem=128G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-115 ##yearly


#¨¨¨¨¨¨!!!!!!!!This script to be executed only after SR_UNet_xx.sh is complete for both bicubic and unet modes. 
#AstahanSh. 

module load python
source diffscaler.sh
module load cdo

cd ../GCM_pipeline

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK


START_YEARS=({1981..2010} {2015..2100})
END_YEARS=({1981..2010} {2015..2100}) #86 years ,,, 30 years for trend pres eval..... 

START_YEAR=${START_YEARS[$SLURM_ARRAY_TASK_ID]}
END_YEAR=${END_YEARS[$SLURM_ARRAY_TASK_ID]}


MODE=${MODE:-ddim}

for ENSEMBLE in EQM_C CDF-t dOTC; do
    echo "($START_YEAR-$END_YEAR) started in mode $MODE for ensemble $ENSEMBLE"
    python SR_downscaling/inference_allframes_eta0_ssp370.py --start_year $START_YEAR --end_year $END_YEAR --mode $MODE --ensemble $ENSEMBLE
    echo "$MODE for RCP45 ($START_YEAR-$END_YEAR) finished for $ENSEMBLE"
done




