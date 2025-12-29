#!/bin/bash
#SBATCH --job-name=dOTC_BC_AllCells
#SBATCH --output=logs/bc/dOTC_BC_AllCells_output-%j.txt
#SBATCH --error=logs/bc/dOTC_BC_AllCells_job_error-%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=3-00:00:00
#SBATCH --mem=512G
#SBATCH --partition=cpu

module load python
source environment.sh

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd BC_methods

#echo "EQM for All Cells started"
#python EQM_allcells.py --n_jobs $SLURM_CPUS_PER_TASK


echo "dOTC for all Cells started"
python dOTC_BC_AllCells.py --n_jobs $SLURM_CPUS_PER_TASK
echo "dOTC for all Cells finished"

#echo "QDM for all Cells started"
#python QDM_BC_AllCells.py --n_jobs $SLURM_CPUS_PER_TASK
#echo "QDM for all Cells finished"