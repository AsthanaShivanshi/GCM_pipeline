import os
import glob
import subprocess
import config
import xarray as xr

CH_BOX = (5, 11, 45, 48)

# Mask file path
PRECIP_MASK_PATH = f"{config.BASE_DIR}/sasthana/Downscaling/Processing_and_Analysis_Scripts/Python_Pipeline_Scripts/precip_mask.nc"

PR_INPUT_DIR = f"{config.MODELS_RUNS_EUROCORDEX_11_RCP85}/pr"
PR_OUTPUT_DIR = f"{config.BASE_DIR}/sasthana/Downscaling/GCM_pipeline/EUROCORDEX_11_RCP8.5/pr_Swiss"

if not os.path.exists(PR_OUTPUT_DIR):
    os.makedirs(PR_OUTPUT_DIR, exist_ok=True)

def process_file(source, outname, oldvar, newvar, mask_path):
    outdir = os.path.dirname(outname)
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)

    step1 = outname.replace(".nc", "_step1.nc")
    # crop
    if not os.path.exists(step1):
        cdo_cmd = [
            "cdo",
            f"sellonlatbox,{CH_BOX[0]},{CH_BOX[1]},{CH_BOX[2]},{CH_BOX[3]}",
            source,
            step1
        ]
        print("Running:", " ".join(cdo_cmd))
        result = subprocess.run(cdo_cmd, capture_output=True, text=True)
        print(result.stdout)
        print(result.stderr)

    # Masking renaming
    if not os.path.exists(outname):
        ds = xr.open_dataset(step1)
        ds = ds.rename({oldvar: newvar})
        mask_ds = xr.open_dataset(mask_path)
        mask = mask_ds["mask"]

        # Align mask to data grid (latlon) using NN interp
        mask_aligned = mask.reindex_like(ds[newvar].isel(time=0), method="nearest")

        # Broadcast mask to all timesteps
        if "time" in ds[newvar].dims and "time" not in mask_aligned.dims:
            mask_broadcast = mask_aligned.expand_dims({"time": ds[newvar].coords["time"]}, axis=0)
        else:
            mask_broadcast = mask_aligned

        ds[newvar] = ds[newvar].where(mask_broadcast)
        ds.to_netcdf(outname, mode="w")
        ds.close()
        print(f"Masked and renamed {oldvar} to {newvar} in {outname}")
    else:
        print(f"Final coarse masked file exists: {outname}, skipping masking.")

    # Delete intermediate file
    if os.path.exists(step1):
        os.remove(step1)
        print(f"Deleted temp file {step1}")

# Loop through all .nc files in pr directory
pr_files = sorted(glob.glob(os.path.join(PR_INPUT_DIR, "*.nc")))

for src in pr_files:
    filename = os.path.basename(src)
    out = os.path.join(PR_OUTPUT_DIR, filename)
    process_file(src, out, "pr", "pr", PRECIP_MASK_PATH)