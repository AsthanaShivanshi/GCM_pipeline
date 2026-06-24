import os
BASE_DIR = os.environ.get("BASE_DIR", "/work/FAC/FGSE/IDYST/tbeucler/downscaling")
DATASETS_TRAINING_DIR= os.environ.get("DATASETS_TRAINING_DIR", f"{BASE_DIR}/sasthana/Downscaling/Downscaling_Models/Dataset_Setup_I_Chronological_12km")
TARGET_DIR= os.environ.get("TARGET_DIR", f"{BASE_DIR}/sasthana/Downscaling/Processing_and_Analysis_Scripts/data_1971_2023/HR_files_full")   
CORDEX_CMIP6= os.environ.get("CORDEX_CMIP6", f"{BASE_DIR}/sasthana/Downscaling/GCM_pipeline/Downscaling/GCM_pipeline/CORDEX-CMIP6/leonardo_work/ICT26_ESP/CORDEX-CMIP6/DD/EUR-12/ICTP")
BC_SCRIPTS= os.environ.get("BC_SCRIPTS", f"{BASE_DIR}/sasthana/Downscaling/GCM_pipeline/BC_methods")
PROCESSED_EUR_12_DIR= os.environ.get("PROCESSED_EUR_12_DIR", f"{BASE_DIR}/sasthana/Downscaling/GCM_pipeline/EUR_12_processed")
SCRATCH_DIR= os.environ.get("SCRATCH_DIR", f"/scratch/sasthana")