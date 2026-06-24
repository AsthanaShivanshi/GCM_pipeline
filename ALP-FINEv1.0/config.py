import os
BASE_DIR = os.environ.get("BASE_DIR", "/work/FAC/FGSE/IDYST/tbeucler/downscaling")
DATASETS_TRAINING_DIR = os.environ.get("DATASETS_TRAINING_DIR", f"{BASE_DIR}/sasthana/Downscaling/Downscaling_Models/Dataset_Setup_I_Chronological_12km")
TARGET_DIR = os.environ.get("TARGET_DIR", f"{BASE_DIR}/sasthana/Downscaling/Processing_and_Analysis_Scripts/data_1971_2023/HR_files_full")
BC_SCRIPTS = os.environ.get("BC_SCRIPTS", f"{BASE_DIR}/sasthana/Downscaling/GCM_pipeline/BC_methods")
SCRATCH_DIR = os.environ.get("SCRATCH_DIR", "/scratch/sasthana")
ALPFINE_DIR = os.environ.get("ALPFINE_DIR", f"{BASE_DIR}/sasthana/Downscaling/GCM_pipeline/ALP-FINEv1.0")

GCM_PIPELINE_DIR = os.environ.get("GCM_PIPELINE_DIR", f"{BASE_DIR}/sasthana/Downscaling/GCM_pipeline")
ALPFINE_COARSE_DIR = os.environ.get("ALPFINE_COARSE_DIR", f"{BASE_DIR}/sasthana/Downscaling/GCM_pipeline/ALP-FINEv1.0/Coarse")
CORDEX_CMIP6 = os.environ.get(
    "CORDEX_CMIP6",
    f"{GCM_PIPELINE_DIR}/CORDEX-CMIP6/leonardo_work/ICT26_ESP/CORDEX-CMIP6/DD/EUR-12/ICTP/",
)