import os
import xarray as xr
import numpy as np
from scipy.interpolate import griddata
import config

# CH bounding box
CH_BOX = (5, 11, 45, 48)


def masking(input_path, mask_path, output_folder, varname):
    try:
        with xr.open_dataset(input_path) as ds:
            lat2d = ds['lat']
            lon2d = ds['lon']

            # Create mask for Swiss bounding box
            swiss_mask = (
                (lon2d >= CH_BOX[0]) & (lon2d <= CH_BOX[1]) &
                (lat2d >= CH_BOX[2]) & (lat2d <= CH_BOX[3])
            )

            # Find the minimal bounding rectangle for the mask
            mask_any = swiss_mask.any(dim='rlon')
            rlat_min = int(mask_any.values.argmax())
            rlat_max = int(len(mask_any) - np.flip(mask_any.values).argmax())

            mask_any = swiss_mask.any(dim='rlat')
            rlon_min = int(mask_any.values.argmax())
            rlon_max = int(len(mask_any) - np.flip(mask_any.values).argmax())

            # Crop to the bounding rectangle
            ds_cropped = ds.isel(
                rlat=slice(rlat_min, rlat_max),
                rlon=slice(rlon_min, rlon_max)
            )

            # Apply mask to the variable (now only a small region)
            ds_cropped[varname] = ds_cropped[varname].where(
                (ds_cropped['lon'] >= CH_BOX[0]) & (ds_cropped['lon'] <= CH_BOX[1]) &
                (ds_cropped['lat'] >= CH_BOX[2]) & (ds_cropped['lat'] <= CH_BOX[3])
            )

            # Optionally drop rlat and rlon if not needed
            # ds_cropped = ds_cropped.drop_vars(['rlat', 'rlon'], errors='ignore')

            output_path = os.path.join(output_folder, os.path.basename(input_path))
            ds_cropped.to_netcdf(output_path)
            print(f"Processed: {input_path} -> {output_path}")
    except Exception as e:
        print(f"Error processing {input_path}: {e}")

VAR_CONFIG = {
    "pr": {
        "input_folder": os.path.join(config.EUROCORDEX_11_DIR, "pr"),
        "mask_path": os.path.join(config.EUROCORDEX_11_DIR, "precip_mask_11km.nc"),
        "output_folder": os.path.join(config.EUROCORDEX_11_DIR, "pr_Swiss"),
    },
    "tas": {
        "input_folder": os.path.join(config.EUROCORDEX_11_DIR, "tas"),
        "mask_path": os.path.join(config.EUROCORDEX_11_DIR, "temp_mask_11km.nc"),
        "output_folder": os.path.join(config.EUROCORDEX_11_DIR, "tas_Swiss"),
    },
    "tasmax": {
        "input_folder": os.path.join(config.EUROCORDEX_11_DIR, "tasmax"),
        "mask_path": os.path.join(config.EUROCORDEX_11_DIR, "temp_mask_11km.nc"),
        "output_folder": os.path.join(config.EUROCORDEX_11_DIR, "tasmax_Swiss"),
    },
    "tasmin": {
        "input_folder": os.path.join(config.EUROCORDEX_11_DIR, "tasmin"),
        "mask_path": os.path.join(config.EUROCORDEX_11_DIR, "temp_mask_11km.nc"),
        "output_folder": os.path.join(config.EUROCORDEX_11_DIR, "tasmin_Swiss"),
    },
}

def process_var(varname, cfg):
    os.makedirs(cfg["output_folder"], exist_ok=True)
    files = [f for f in os.listdir(cfg["input_folder"]) if f.endswith(".nc")]
    for fname in files:
        input_path = os.path.join(cfg["input_folder"], fname)
        masking(input_path, cfg["mask_path"], cfg["output_folder"], varname)


if __name__ == "__main__":
    for varname, cfg in VAR_CONFIG.items():
        process_var(varname, cfg)
