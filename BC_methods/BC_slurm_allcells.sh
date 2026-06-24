#!/bin/bash
#SBATCH --job-name=EQM_pr_tas
#SBATCH --output=logs/bc/EQM/EQM_pr_tas_output-%j.txt
#SBATCH --error=logs/bc/EQM/EQM_pr_tas_job_error-%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00
#SBATCH --mem=128G
#SBATCH --partition=cpu

module load python
source ../environment.sh

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd BC_methods

echo "EQM for All Cells started"
python EQM_allcells.py --n_jobs $SLURM_CPUS_PER_TASK
echo "EQM for All Cells finished"


