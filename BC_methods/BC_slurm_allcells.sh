#!/bin/bash
#SBATCH --job-name=EQM_BC_taspr
#SBATCH --output=logs/bc/EQM_BC_output-%j.txt
#SBATCH --error=logs/bc/EQM_BC_job_error-%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=3-00:00:00
#SBATCH --mem=128G
#SBATCH --partition=cpu

module load python
source ../diffscaler.sh
cd ../GCM_pipeline/BC_methods

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK


echo "EQM for All Cells started"
python EQM_BC_allcells.py --n_jobs $SLURM_CPUS_PER_TASK
echo "EQM for All Cells finished"


