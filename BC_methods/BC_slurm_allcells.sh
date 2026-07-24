#!/bin/bash
#SBATCH --job-name=dOTC
#SBATCH --output=logs/bc/dOTC-%j.log
#SBATCH --error=logs/bc/dOTC-%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --mem=256G
#SBATCH --partition=cpu

module load python
source ../diffscaler.sh
cd ../GCM_pipeline/BC_methods

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK



#echo "EQM Coarse for All Cells started"
#python EQM_BC_allcells.py --n_jobs $SLURM_CPUS_PER_TASK 

#echo "EQM for all cells finished"


echo "dOTC Coarse for All Cells started"
python dOTC_BC_allcells.py --n_jobs $SLURM_CPUS_PER_TASK
echo "dOTC Coarse for all cells finished"


#echo "CDFt Coarse for All Cells started"
#python CDFt_BC_allcells.py --n_jobs $SLURM_CPUS_PER_TASK 

#echo "CDFt for all cells finished"



