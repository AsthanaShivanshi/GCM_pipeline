#!/bin/bash
#SBATCH --job-name=EUR_11_Preprocess
#SBATCH --output=logs/preprocess_EUR_11/job_output-%j.txt
#SBATCH --error=logs/preprocess_EUR_11/job_error-%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=10:00:00
#SBATCH --mem=512G
#SBATCH --partition=cpu
##SBATCH --partition=gpu

source environment.sh
export PYTHONPATH="$PROJECT_DIR"
mkdir -p logs/preprocess_EUR_11

cd "$BASE_DIR/sasthana/Downscaling/GCM_pipeline"
python EUROCORDEX_11/model_outputs_preprocessing.py