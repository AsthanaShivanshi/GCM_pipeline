import os
import xarray as xr
import numpy as np
import config
import subprocess

CH_BOX = (5, 11, 45, 48)

def remap_mask_to_model(mask_path, model_path, remapped_mask_path):
    if not os.path.exists(remapped_mask_path):
        cmd = [
            "cdo",
            f"remapnn,{model_path}",
            mask_path,
            remapped_mask_path
        ]
        print("Running:", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        print(result.stderr)
        if result.returncode != 0 or not os.path.exists(remapped_mask_path):
            raise RuntimeError(f"CDO remapnn failed for {mask_path}")

def masking(input_path, mask_path, output_folder, varname, mask_varname):
    try:

        remapped_mask_path = mask_path.replace(".nc", "_remapped.nc")
        remap_mask_to_model(mask_path, input_path, remapped_mask_path)

        with xr.open_dataset(input_path) as ds, xr.open_dataset(remapped_mask_path) as mask_ds:
            lat2d = ds['lat'].values
            lon2d = ds['lon'].values

            bbox_mask = (
                (lon2d >= CH_BOX[0]) & (lon2d <= CH_BOX[1]) &
                (lat2d >= CH_BOX[2]) & (lat2d <= CH_BOX[3])
            )

            mask_data = mask_ds[mask_varname].isel(time=0) if 'time' in mask_ds[mask_varname].dims else mask_ds[mask_varname]
            mask_non_nan = ~np.isnan(mask_data.values)

            model_mask = bbox_mask & mask_non_nan

            ds[varname] = ds[varname].where(model_mask)

            valid_idx = np.where(model_mask)
            if valid_idx[0].size > 0:
                rlat_min, rlat_max = valid_idx[0].min(), valid_idx[0].max() + 1
                rlon_min, rlon_max = valid_idx[1].min(), valid_idx[1].max() + 1
                ds = ds.isel(rlat=slice(rlat_min, rlat_max), rlon=slice(rlon_min, rlon_max))

            output_path = os.path.join(output_folder, os.path.basename(input_path))
            ds.to_netcdf(output_path)
            print(f"Processed: {input_path} -> {output_path}")

        if os.path.exists(remapped_mask_path):
            os.remove(remapped_mask_path)

    except Exception as e:
        print(f"Error processing {input_path}: {e}")


VAR_CONFIG = {
    "pr": {
        "input_folder": os.path.join(config.EUROCORDEX_11_DIR, "pr"),
        "mask_path": os.path.join(config.MODEL_RUNS_DIR, "precip_MPI-CSC-REMO2009_MPI-M-MPI-ESM-LR_rcp85_1971-2099/precip_r01_coarse_masked.nc"),
        "mask_varname": "precip",
        "output_folder": os.path.join(config.EUROCORDEX_11_DIR, "pr_Swiss"),
    },
    "tas": {
        "input_folder": os.path.join(config.EUROCORDEX_11_DIR, "tas"),
        "mask_path": os.path.join(config.DATASETS_TRAINING_DIR, "TabsD_step2_coarse.nc"),
        "mask_varname": "TabsD",
        "output_folder": os.path.join(config.EUROCORDEX_11_DIR, "tas_Swiss"),
    },
    "tasmax": {
        "input_folder": os.path.join(config.EUROCORDEX_11_DIR, "tasmax"),
        "mask_path": os.path.join(config.DATASETS_TRAINING_DIR, "TabsD_step2_coarse.nc"),
        "mask_varname": "TabsD",
        "output_folder": os.path.join(config.EUROCORDEX_11_DIR, "tasmax_Swiss"),
    },
    "tasmin": {
        "input_folder": os.path.join(config.EUROCORDEX_11_DIR, "tasmin"),
        "mask_path": os.path.join(config.DATASETS_TRAINING_DIR, "TabsD_step2_coarse.nc"),
        "mask_varname": "TabsD",
        "output_folder": os.path.join(config.EUROCORDEX_11_DIR, "tasmin_Swiss"),
    },
}

def process_var(varname, cfg):
    os.makedirs(cfg["output_folder"], exist_ok=True)
    files = [f for f in os.listdir(cfg["input_folder"]) if f.endswith(".nc")]
    for fname in files:
        input_path = os.path.join(cfg["input_folder"], fname)
        masking(input_path, cfg["mask_path"], cfg["output_folder"], varname, cfg["mask_varname"])

if __name__ == "__main__":
    for varname, cfg in VAR_CONFIG.items():
        process_var(varname, cfg)