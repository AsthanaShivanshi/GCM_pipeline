#!/bin/bash
#SBATCH --job-name=dOTC_Coarse_BC
#SBATCH --output=logs/bc/dOTC_BC_output-%j.log
#SBATCH --error=logs/bc/dOTC_BC_error-%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02-00:00:00
#SBATCH --mem=256G
#SBATCH --partition=cpu

module load python
source ../diffscaler.sh
cd ../GCM_pipeline/BC_methods

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "dOTC Coarse for All Cells started"
python dOTC_BC_allcells.py --n_jobs $SLURM_CPUS_PER_TASK
echo "dOTC Coarse for all cells finished"