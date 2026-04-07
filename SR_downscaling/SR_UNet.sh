#!/bin/bash
#SBATCH --job-name=UNet_DDIM_SR_RCP26_BC_allcells
#SBATCH --output=logs/bc/UNet_SR/UNet_DDIM_SR_pr_tas_RCP26_BC_AllCells_output-%j.txt
#SBATCH --error=logs/bc/UNet_SR/UNet_DDIM_SR_pr_tas_RCP26_BC_AllCells_job_error-%j.txt
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

echo "UNet_SR and DDIM for all Cells for RCP26 (1971-2099) started"
python SR_downscaling/inference_allframes_eta0_RCP26.py --n_jobs $SLURM_CPUS_PER_TASK
echo "UNet_SR and DDIM for all Cells for RCP26 (1971-2099) finished"

