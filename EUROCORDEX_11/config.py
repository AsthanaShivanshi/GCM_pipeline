import os
BASE_DIR = os.environ.get("BASE_DIR", "/work/FAC/FGSE/IDYST/tbeucler/downscaling")
DATASETS_TRAINING_DIR= os.environ.get("DATASETS_TRAINING_DIR", f"{BASE_DIR}/sasthana/Downscaling/Downscaling_Models/Dataset_Setup_I_Chronological_12km")
TARGET_DIR= os.environ.get("TARGET_DIR", f"{BASE_DIR}/sasthana/Downscaling/Processing_and_Analysis_Scripts/data_1971_2023/HR_files_full")   
GCM_PIPELINE_DIR= os.environ.get("GCM_PIPELINE_DIR", f"{BASE_DIR}/sasthana/Downscaling/GCM_pipeline")   
EUROCORDEX_11_DIR= os.environ.get("EUROCORDEX_11_DIR", f"{BASE_DIR}/sasthana/Downscaling/GCM_pipeline/EUROCORDEX_11")
MODEL_RUNS_DIR= os.environ.get("MODEL_RUNS_DIR", f"{BASE_DIR}/sasthana/Downscaling/Downscaling_Models/Model_Runs")
