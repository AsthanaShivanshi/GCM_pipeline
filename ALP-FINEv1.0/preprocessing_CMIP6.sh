#!/bin/bash
#SBATCH --job-name=CMIP6_Preprocessing
#SBATCH --output=ALP-FINEv1.0/logs/preprocessing_CMIP6.log
#SBATCH --error=ALP-FINEv1.0/logs/preprocessing_CMIP6.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --mem=128G
#SBATCH --partition=cpu


#Run from GP

source environment.sh
module load cdo
export PYTHONPATH="$PROJECT_DIR"
mkdir -p ALP-FINEv1.0/logs

python ALP-FINEv1.0/model_runs_preprocessing.py