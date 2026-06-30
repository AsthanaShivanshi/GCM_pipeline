#!/bin/bash
#SBATCH --job-name=CDFt_Coarse_BC
#SBATCH --output=logs/bc/CDFt_BC_output-%j.log
#SBATCH --error=logs/bc/CDFt_BC_error-%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01-00:00:00
#SBATCH --mem=256G
#SBATCH --partition=cpu

module load python
source ../diffscaler.sh
cd ../GCM_pipeline/BC_methods

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "CDFt Coarse for All Cells started"
python CDFt_BC_allcells.py --n_jobs $SLURM_CPUS_PER_TASK
echo "CDFt Coarse for all cells finished"