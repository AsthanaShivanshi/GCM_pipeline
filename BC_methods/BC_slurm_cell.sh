#!/bin/bash
#SBATCH --job-name=Cities_BC_multiple_vars
#SBATCH --output=logs/bc/BC_tabs_output-%j.txt
#SBATCH --error=logs/bc/BC_tabs_job_error-%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --mem=128G
#SBATCH --partition=cpu

module load python
source environment.sh


export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd BC_methods

#echo "EQM for single cell started"
#python EQM_single_cell.py --city Zurich --lat 47.3769 --lon 8.5417
#python EQM_single_cell.py --city Locarno --lat 46.1670 --lon 8.7943
python EQM_single_cell.py --city Geneva --lat 46.2044 --lon 6.1432

#echo "DOTC for single cell started"
#python DOTC_BC_single_cell.py --city Zurich --lat 47.3769 --lon 8.5417
#python DOTC_BC_single_cell.py --city Locarno --lat 46.1670 --lon 8.7943
python DOTC_BC_single_cell.py --city Geneva --lat 46.2044 --lon 6.1432

#echo "QDM for single cell started"
#python QDM_single_cell.py --city Zurich --lat 47.3769 --lon 8.5417
#python QDM_single_cell.py --city Locarno --lat 46.1670 --lon 8.7943
python QDM_single_cell.py --city Geneva --lat 46.2044 --lon 6.1432

#echo "Residual CDF and CvM test"
#python CvM_Bias_of_CDFs.py --city Zurich --lat 47.3769 --lon 8.5417
#python CvM_Bias_of_CDFs.py --city Locarno --lat 46.1670 --lon 8.7943
python CvM_Bias_of_CDFs.py --city Geneva --lat 46.2044 --lon 6.1432