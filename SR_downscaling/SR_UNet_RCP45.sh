#!/bin/bash
#SBATCH --job-name=UNet_SR_RCP45_BC_allcells
#SBATCH --output=logs/SR/UNet_SR_RCP45_BC_AllCells_output-%A_%a.log
#SBATCH --error=logs/SR/UNet_SR_RCP45_BC_AllCells_job_error-%A_%a.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --mem=128G
#SBATCH --partition=gpu   ##OR gpu-h100
#SBATCH --array=0-12 ##13blocks for 10 years each 
#SBATCH --gres=gpu:1

module load python
source diffscaler.sh
module load cdo

cd ../GCM_pipeline

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

START_YEARS=(1971 1981 1991 2001 2011 2021 2031 2041 2051 2061 2071 2081 2091)
END_YEARS=(1980 1990 2000 2010 2020 2030 2040 2050 2060 2070 2080 2090 2100)

START_YEAR=${START_YEARS[$SLURM_ARRAY_TASK_ID]}
END_YEAR=${END_YEARS[$SLURM_ARRAY_TASK_ID]}

MODE=${MODE:-unet} #modes possible : bicubic, unet, ......ddim is in a separate .sh script...,,.. should be run sequentially. 

# sequential


for ENSEMBLE in EQM CDFT dOTC; do
    echo "($START_YEAR-$END_YEAR) started in mode $MODE for ensemble $ENSEMBLE"
    python SR_downscaling/inference_allframes_eta0_RCP45.py --start_year $START_YEAR --end_year $END_YEAR --mode $MODE --ensemble $ENSEMBLE
    echo "$MODE for RCP45 ($START_YEAR-$END_YEAR) finished for $ENSEMBLE"
done




JOBID=$SLURM_JOB_ID

echo "Summary of SLURM array job $JOBID:"
sacct -j ${JOBID} --format=JobID,JobName,Elapsed,MaxRSS,AllocCPUS,ReqMem,State

