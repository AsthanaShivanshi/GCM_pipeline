import os
import xarray as xr
import numpy as np
from scipy.interpolate import griddata
import config

# CH bounding box
CH_BOX = (5, 11, 45, 48)


def masking(input_path, mask_path, output_folder, varname):
    try:
        ds = xr.open_dataset(input_path)
        mask_ds = xr.open_dataset(mask_path)

        # Use 2D lon/lat variables for masking
        lon2d = ds['lon'].values
        lat2d = ds['lat'].values
        mask_ch = (
            (lon2d >= CH_BOX[0]) & (lon2d <= CH_BOX[1]) &
            (lat2d >= CH_BOX[2]) & (lat2d <= CH_BOX[3])
        )

        # Mask the data
        data_array = ds[varname].values
        masked_data = np.where(mask_ch, data_array, np.nan)

        ds[varname].values = masked_data
        output_path = os.path.join(output_folder, os.path.basename(input_path))
        ds.to_netcdf(output_path)
        ds.close()
        mask_ds.close()
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
