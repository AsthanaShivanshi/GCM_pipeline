#!/bin/bash
#SBATCH --job-name=DDIM_SR_RCP85_BC_allcells
#SBATCH --output=logs/SR/DDIM_SR_SR_pr_tas_RCP85_BC_AllCells_output-%A_%a.txt
#SBATCH --error=logs/SR/DDIM_SR_SR_pr_tas_RCP85_BC_AllCells_job_error-%A_%a.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=03-00:00:00
#SBATCH --mem=128G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-5  ##decadal jobs... 

module load python
source diffscaler.sh
module load cdo

cd ../GCM_pipeline

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

START_YEARS=(1971 1991 2011 2031 2051 2071)
END_YEARS=(1990 2010 2030 2050 2070 2099)

START_YEAR=${START_YEARS[$SLURM_ARRAY_TASK_ID]}
END_YEAR=${END_YEARS[$SLURM_ARRAY_TASK_ID]}

MODE=${MODE:-ddim}

# sequential


for ENSEMBLE in EQM CDFT dOTC; do
    echo "($START_YEAR-$END_YEAR) started in mode $MODE for ensemble $ENSEMBLE"
    python SR_downscaling/inference_allframes_eta0_RCP85.py --start_year $START_YEAR --end_year $END_YEAR --mode $MODE --ensemble $ENSEMBLE
    echo "$MODE for RCP85 ($START_YEAR-$END_YEAR) finished for $ENSEMBLE"
done



JOBID=$SLURM_JOB_ID

echo "Summary of SLURM array job $JOBID:"


sacct -j ${JOBID} --format=JobID,JobName,Elapsed,MaxRSS,AllocCPUS,ReqMem,State

