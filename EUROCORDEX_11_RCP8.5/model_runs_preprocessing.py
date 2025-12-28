import os
import glob
import subprocess
import config
import xarray as xr

CH_BOX = (5, 11, 45, 48)

# Mask file path
TEMP_MASK_PATH = f"{config.BASE_DIR}/sasthana/Downscaling/Downscaling_Models/temp_mask_12km.nc"

TEMP_INPUT_DIR = f"{config.MODELS_RUNS_EUROCORDEX_11_RCP85}/tas"
TEMP_OUTPUT_DIR = f"{config.BASE_DIR}/sasthana/Downscaling/GCM_pipeline/EUROCORDEX_11_RCP8.5/tas_Swiss"

if not os.path.exists(TEMP_OUTPUT_DIR):
    os.makedirs(TEMP_OUTPUT_DIR, exist_ok=True)
def process_file(source, outname, oldvar, newvar, mask_path):
    outdir = os.path.dirname(outname)
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)

    step1 = outname.replace(".nc", "_step1.nc")
    step2 = outname.replace(".nc", "_step2.nc")

    # Cropping to CH bounding box
    if not os.path.exists(step1):
        cdo_cmd_crop = [
            "cdo",
            f"sellonlatbox,{CH_BOX[0]},{CH_BOX[1]},{CH_BOX[2]},{CH_BOX[3]}",
            source,
            step1
        ]
        print("Running:", " ".join(cdo_cmd_crop))
        result = subprocess.run(cdo_cmd_crop, capture_output=True, text=True)
        print(result.stdout)
        print(result.stderr)

    # Remapnn to align with mask
    if not os.path.exists(step2):
        cdo_cmd_remap = [
            "cdo",
            f"remapnn,{mask_path}",
            step1,
            step2
        ]
        print("Running:", " ".join(cdo_cmd_remap))
        result = subprocess.run(cdo_cmd_remap, capture_output=True, text=True)
        print(result.stdout)
        print(result.stderr)

    # Masking and renaming
    if not os.path.exists(outname):
        ds = xr.open_dataset(step2)
        ds = ds.rename({oldvar: newvar})
        mask_ds = xr.open_dataset(mask_path)
        mask = mask_ds["TabsD"]

        ds[newvar] = ds[newvar].where(mask)
        ds.to_netcdf(outname, mode="w")
        ds.close()
        print(f"Masked and renamed {oldvar} to {newvar} in {outname}")
    else:
        print(f"Final coarse masked file exists: {outname}, skipping masking.")

    # Delete intermediate files
    if os.path.exists(step1):
        os.remove(step1)
        print(f"Deleted temp file {step1}")
    if os.path.exists(step2):
        os.remove(step2)
        print(f"Deleted temp file {step2}")


#Looping
temp_files = sorted(glob.glob(os.path.join(TEMP_INPUT_DIR, "*.nc")))

for src in temp_files:
    filename = os.path.basename(src)
    out = os.path.join(TEMP_OUTPUT_DIR, filename)
    process_file(src, out, "tas", "tas", TEMP_MASK_PATH)