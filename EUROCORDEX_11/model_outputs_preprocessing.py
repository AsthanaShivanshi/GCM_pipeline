import os
import xarray as xr
import numpy as np
from scipy.interpolate import griddata

# Swiss bounding box (lon_min, lon_max, lat_min, lat_max)
CH_BOX = (5, 11, 45, 48)

def masking(input_path, mask_path, output_folder, varname):
    fname = os.path.basename(input_path)
    outname = os.path.join(output_folder, fname.replace(".nc", "_Swiss_masked_11km.nc"))

    if os.path.exists(outname):
        print(f"Masked file exists: {outname}, skipping.")
        return

    with xr.open_dataset(input_path) as ds:
        lat2d = ds["lat"].values
        lon2d = ds["lon"].values

        # Swiss boiundign box
        swiss_mask = (
            (lon2d >= CH_BOX[0]) & (lon2d <= CH_BOX[1]) &
            (lat2d >= CH_BOX[2]) & (lat2d <= CH_BOX[3])
        )

        # Crop rlat/rlon to Swiss domain
        rlat_idx, rlon_idx = np.where(swiss_mask)
        rlat_sel = np.unique(ds['rlat'].values[rlat_idx])
        rlon_sel = np.unique(ds['rlon'].values[rlon_idx])
        ds = ds.sel(rlat=rlat_sel, rlon=rlon_sel)
        lat2d = ds["lat"].values
        lon2d = ds["lon"].values

        with xr.open_dataset(mask_path) as mask_ds:
            mask = mask_ds["mask"].values
            mask_lat = mask_ds["lat"].values
            mask_lon = mask_ds["lon"].values

            # Flatten mask grid for interpolation
            points = np.column_stack([mask_lat.ravel(), mask_lon.ravel()])
            values = mask.ravel()

            # Interpolate mask to cropped data grid
            mask_on_data_grid = griddata(points, values, (lat2d, lon2d), method='nearest')

        # Broadcast mask to match data shape (time, rlat, rlon)
        if "time" in ds[varname].dims:
            mask_broadcast = np.broadcast_to(mask_on_data_grid, ds[varname].shape)
        else:
            mask_broadcast = mask_on_data_grid

        # Apply mask
        ds[varname] = ds[varname].where(mask_broadcast)

        os.makedirs(output_folder, exist_ok=True)
        ds.to_netcdf(outname, mode="w")

import config

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

for varname, cfg in VAR_CONFIG.items():
    os.makedirs(cfg["output_folder"], exist_ok=True)
    for fname in os.listdir(cfg["input_folder"]):
        if fname.endswith(".nc"):
            input_path = os.path.join(cfg["input_folder"], fname)
            masking(input_path, cfg["mask_path"], cfg["output_folder"], varname)