#!/bin/bash
#SBATCH --job-name=EUR_11_Preprocessing_pr_rcp85
#SBATCH --output=logs/preprocess_pr_rcp85_job_output-%j.txt
#SBATCH --error=logs/preprocess_pr_rcp85_job_error-%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=15:00:00
#SBATCH --mem=256G
#SBATCH --partition=cpu
##SBATCH --partition=gpu

source environment.sh
module load cdo
export PYTHONPATH="$PROJECT_DIR"
mkdir -p logs

cd "$BASE_DIR/sasthana/Downscaling/GCM_pipeline"
python EUROCORDEX_11_RCP8.5/model_runs_preprocessing.py