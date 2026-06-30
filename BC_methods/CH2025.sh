#!/bin/bash
#SBATCH --job-name=CH2025
#SBATCH --output=logs/bc/CH2025_BC_output-%j.log
#SBATCH --error=logs/bc/CH2025_BC_error-%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=03-00:00:00
#SBATCH --mem=256G
#SBATCH --partition=cpu

module load python
source ../diffscaler.sh
cd ../GCM_pipeline/BC_methods

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "CH2025 for All Cells started"
python CH2025_method_EQM_allcells.py --n_jobs "$SLURM_CPUS_PER_TASK"
echo "CH2025 Coarse for all cells finished"