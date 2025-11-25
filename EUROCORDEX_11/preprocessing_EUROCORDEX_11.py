import os
import subprocess
import config
import xarray as xr

CH_BOX = (5, 11, 45, 48)

# Mask file paths
TEMP_MASK_PATH = f"{config.BASE_DIR}/sasthana/Downscaling/Processing_and_Analysis_Scripts/Python_Pipeline_Scripts/temp_mask.nc"
PRECIP_MASK_PATH = f"{config.BASE_DIR}/sasthana/Downscaling/Processing_and_Analysis_Scripts/Python_Pipeline_Scripts/precip_mask.nc"

# HR grid paths
TEMP_HR_GRID = f"{config.DATASETS_COMBINED_DIR}/combined_temp_target_test_chronological_scaled.nc"
PRECIP_HR_GRID = f"{config.DATASETS_COMBINED_DIR}/combined_precip_target_test_chronological_scaled.nc"

def process_file(source, target, outname, oldvar, newvar, mask_path):
    outdir = os.path.dirname(outname)
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)

    step1 = outname.replace(".nc", "_step1.nc")
    step2 = outname.replace(".nc", "_step2.nc")

    # remapbic
    if not os.path.exists(step1):
        cdo_cmd1 = [
            "cdo",
            f"-remapbic,{target}",
            source,
            step1
        ]
        print("Running:", " ".join(cdo_cmd1))
        result1 = subprocess.run(cdo_cmd1, capture_output=True, text=True)
        print(result1.stdout)
        print(result1.stderr)
        if result1.returncode != 0 or not os.path.exists(step1):
            print(f"CDO remapbic failed for {source}")
            return
    else:
        print(f"Step1 file exists: {step1}, skipping remapbic.")

    # crop
    if not os.path.exists(step2):
        cdo_cmd2 = [
            "cdo",
            f"sellonlatbox,{CH_BOX[0]},{CH_BOX[1]},{CH_BOX[2]},{CH_BOX[3]}",
            step1,
            step2
        ]
        print("Running:", " ".join(cdo_cmd2))
        result2 = subprocess.run(cdo_cmd2, capture_output=True, text=True)
        print(result2.stdout)
        print(result2.stderr)
        if result2.returncode != 0 or not os.path.exists(step2):
            print(f"CDO sellonlatbox failed for {step1}")
            os.remove(step1)
            return
    else:
        print(f"Step2 file exists: {step2}, skipping crop.")

    # Masking renaming
    if not os.path.exists(outname):
        ds = xr.open_dataset(step2)
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

# Interpolating to HR using obs grid
def interpolate_to_HR(coarse_file, hr_grid, out_hr_file, varname):
    if not os.path.exists(out_hr_file):
        cdo_cmd = [
            "cdo",
            f"-remapbic,{hr_grid}",
            coarse_file,
            out_hr_file
        ]
        print("Running HR interpolation:", " ".join(cdo_cmd))
        result = subprocess.run(cdo_cmd, capture_output=True, text=True)
        print(result.stdout)
        print(result.stderr)
        if result.returncode != 0 or not os.path.exists(out_hr_file):
            print(f"CDO HR interpolation failed for {coarse_file}")
            return
        print(f"Saved HR interpolated file: {out_hr_file}")
    else:
        print(f"HR file exists: {out_hr_file}, skipping HR interpolation.")

pairs = [
    (
        f"{config.MODELS_DIR}/pr_day_EUR-11_MPI-CSC-REMO2009_MPI-M-MPI-ESM-LR_r2i1p1_rcp85_1971-2099.nc",
        f"{config.DATASETS_COMBINED_DIR}/precip_test_step2_coarse.nc",
        f"{config.MODELS_DIR}/precip_r02_coarse_masked.nc",
        "pr", "precip", PRECIP_MASK_PATH, PRECIP_HR_GRID
    ),
    (
        f"{config.MODELS_DIR}/tas_day_EUR-11_MPI-CSC-REMO2009_MPI-M-MPI-ESM-LR_r2i1p1_rcp85_1971-2099.nc",
        f"{config.DATASETS_COMBINED_DIR}/temp_test_step2_coarse.nc",
        f"{config.MODELS_DIR}/temp_r02_coarse_masked.nc",
        "tas", "temp", TEMP_MASK_PATH, TEMP_HR_GRID
    ),
    (
        f"{config.MODELS_DIR}/tasmax_day_EUR-11_MPI-CSC-REMO2009_MPI-M-MPI-ESM-LR_r2i1p1_rcp85_1971-2099.nc",
        f"{config.DATASETS_COMBINED_DIR}/tmax_test_step2_coarse.nc",
        f"{config.MODELS_DIR}/tmax_r02_coarse_masked.nc",
        "tasmax", "tmax", TEMP_MASK_PATH, TEMP_HR_GRID
    ),
    (
        f"{config.MODELS_DIR}/tasmin_day_EUR-11_MPI-CSC-REMO2009_MPI-M-MPI-ESM-LR_r2i1p1_rcp85_1971-2099.nc",
        f"{config.DATASETS_COMBINED_DIR}/tmin_test_step2_coarse.nc",
        f"{config.MODELS_DIR}/tmin_r02_coarse_masked.nc",
        "tasmin", "tmin", TEMP_MASK_PATH, TEMP_HR_GRID
    )
]

for src, tgt, out, oldvar, newvar, mask_path, hr_grid in pairs:
    step1 = out.replace(".nc", "_step1.nc")
    step2 = out.replace(".nc", "_step2.nc")
    process_file(src, tgt, out, oldvar, newvar, mask_path)
    out_hr = out.replace("_coarse_masked.nc", "_HR_masked.nc")

    
    interpolate_to_HR(out, hr_grid, out_hr, newvar)
    # Now delete intermediate files
    if os.path.exists(step1):
        os.remove(step1)
    if os.path.exists(step2):
        os.remove(step2)
    print(f"Deleted temp files {step1} and {step2}")