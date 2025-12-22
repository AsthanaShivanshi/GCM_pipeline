import os
import xarray as xr
import numpy as np
from scipy.interpolate import griddata
import config

# CH bounding box
CH_BOX = (5, 11, 45, 48)


def masking(input_path, mask_path, output_folder, varname):
    try:
        with xr.open_dataset(input_path) as ds, xr.open_dataset(mask_path) as mask_ds:
            # Get 2D lat/lon
            lat2d = ds['lat']
            lon2d = ds['lon']

            # Create mask for Swiss bounding box
            swiss_mask = (
                (lon2d >= CH_BOX[0]) & (lon2d <= CH_BOX[1]) &
                (lat2d >= CH_BOX[2]) & (lat2d <= CH_BOX[3])
            )

            # Find indices where mask is True
            idx = np.where(swiss_mask)
            if len(idx[0]) == 0 or len(idx[1]) == 0:
                print(f"No grid points in Swiss bounding box for {input_path}")
                return

            # Get unique indices for cropping
            rlat_idx = np.unique(idx[0])
            rlon_idx = np.unique(idx[1])

            # Crop dataset to Swiss bounding box using isel
            ds_crop = ds.isel(rlat=rlat_idx, rlon=rlon_idx)

            # Assign E and N from mask (assumes mask has dims E, N)
            # Interpolate mask E/N grid to cropped lat/lon grid if needed
            if {'E', 'N'}.issubset(mask_ds.dims):
                # Assign as coordinates
                ds_crop = ds_crop.assign_coords(
                    E=(('rlat', 'rlon'), mask_ds['E'].values),
                    N=(('rlat', 'rlon'), mask_ds['N'].values)
                )
            else:
                print(f"Mask file does not have E and N dimensions for {input_path}")

            # Drop rlat and rlon dimensions if you don't need them
            ds_crop = ds_crop.drop_vars(['rlat', 'rlon'])

            # Save output
            output_path = os.path.join(output_folder, os.path.basename(input_path))
            ds_crop.to_netcdf(output_path)
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
