#!/bin/bash
module load python
source ../diffscaler.sh
cd ../GCM_pipeline/BC_methods

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK



echo "dOTC for All Cells started"
python dOTC_BC_allcells.py --n_jobs $SLURM_CPUS_PER_TASK
echo "dOTC for All Cells finished"
