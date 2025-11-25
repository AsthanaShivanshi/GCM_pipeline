import os
import xarray as xr
import config

# Swiss bounding box
CH_BOX = (5, 11, 45, 48)


def masking(input_path, mask_path, output_folder, varname):
    fname = os.path.basename(input_path)
    outname = os.path.join(output_folder, fname.replace(".nc", "_Swiss_masked_11km.nc"))

    if os.path.exists(outname):
        print(f"Masked file exists: {outname}, skipping.")
        return

    with xr.open_dataset(input_path) as ds:
        lat2d = ds["lat"]
        lon2d = ds["lon"]
        mask_box = (
            (lon2d >= CH_BOX[0]) & (lon2d <= CH_BOX[1]) &
            (lat2d >= CH_BOX[2]) & (lat2d <= CH_BOX[3])
        )
        ds[varname] = ds[varname].where(mask_box)

        with xr.open_dataset(mask_path) as mask_ds:
            mask = mask_ds["mask"]
            mask_aligned = mask.reindex_like(ds[varname].isel(time=0), method="nearest")

        if "time" in ds[varname].dims and "time" not in mask_aligned.dims:
            mask_broadcast = mask_aligned.expand_dims({"time": ds[varname].coords["time"]}, axis=0)
        else:
            mask_broadcast = mask_aligned

        ds[varname] = ds[varname].where(mask_broadcast)

        valid_rlat = ds["rlat"].where(mask_aligned.sum(dim="rlon") > 0, drop=True)
        valid_rlon = ds["rlon"].where(mask_aligned.sum(dim="rlat") > 0, drop=True)
        ds = ds.sel(rlat=valid_rlat, rlon=valid_rlon)

        os.makedirs(output_folder, exist_ok=True)
        ds.to_netcdf(outname, mode="w")


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